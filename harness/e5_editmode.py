#!/usr/bin/env python3
"""
E5 — diff-only vs whole-file, ACROSS TURNS.

The last open design question. Single-turn data in this repo favours anchor edits
(3x cheaper, and a truncated whole-file reply can destroy the file). Aider's published
benchmarks find the opposite for multi-turn: similar single-turn correctness, but
whole-file *more stable* across turns. E4 used anchor edits exclusively, so it did not
settle it.

Same task and same structured brief as E4's winning arm. The ONLY difference is the
edit verb the model is given:

  anchor  EDIT <path> + SEARCH/REPLACE block   (E4's behaviour, 9 turns, solved)
  whole   WRITE <path> + the complete new file

Measured: solved, turns, context, and total tokens -- plus how often a whole-file
write truncates or mangles the file, which is the failure mode anchor edits cannot have.

Original E4 docstring follows.

E4 — does structured handoff cut context?

The hypothesis this whole hybrid design rests on. From the experiment log:

    Session 4 reached ~70K before its first edit. Target: under 10K.
    This is the core hypothesis behind the whole hybrid design -- if structured
    handoff does not dramatically cut context, reconsider the plan.

Two arms, same model, same repo, same bugs. Only the opening instruction differs:

  COLD   "another agent worked on this, please continue, it was on test phase"
  BRIEF  failing test names, the files involved, acceptance criteria, what was tried

Measured: context (prompt tokens) at FIRST EDIT, peak context, turns, tool calls
before first edit, and whether the tests actually end up green.

SAFETY: the model never executes shell. It gets a fixed verb set confined to the
repo copy -- LIST / READ / TEST / EDIT / DONE. No arbitrary commands, matching the
project rule about never auto-running model-suggested commands.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx_server as mx

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "e4_workspace"
PRISTINE = WS / "pristine"
RESULTS = ROOT / "results" / "results-e5.json"

MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-5bit"
PORT = 8097
MAX_TURNS = 40
MAX_TOKENS = 1200          # per assistant turn
CONTEXT_ABORT = 90000      # stop before we hit the memory wall measured earlier

PROTOCOL_HEAD = """You are fixing failing tests in a React + TypeScript project.

You can ONLY use these commands. Emit EXACTLY ONE command per reply, nothing else.

LIST <dir>
    List files in a directory, relative to the project root. Example: LIST src/utils

READ <path>
    Print a file. Example: READ src/utils/weatherUtils.ts

TEST
    Run the test suite and show the result.

{EDIT_VERB}
DONE
    Only when the full test suite passes.

Rules:
- One command per reply. No prose, no explanation, no markdown fences.
- Do not edit test files. Fix the source.
"""

ANCHOR_VERB = """EDIT <path>
<<<<<<< SEARCH
(exact lines to find, copied verbatim from the file)
=======
(replacement lines)
>>>>>>> REPLACE
    The SEARCH text must match the file exactly.

"""

WHOLE_VERB = """WRITE <path>
(the complete new contents of the file, every line of it)
    This REPLACES the entire file. Anything you leave out is deleted.

"""

PROTOCOL_ANCHOR = PROTOCOL_HEAD.replace("{EDIT_VERB}", ANCHOR_VERB)
PROTOCOL_WHOLE = PROTOCOL_HEAD.replace("{EDIT_VERB}", WHOLE_VERB)

COLD = """In the project directory there is a project another agent worked on.
Please continue on the task, it was on test phase."""

BRIEF = """Continue a previous agent's work on this project. Here is the handoff.

STATE: 3 tests are failing out of 26. The rest pass.

FAILING TESTS:
  1. src/utils/weatherUtils.test.ts > convertTemperature > should convert Celsius to Fahrenheit
     expected 32, received 30
  2. src/utils/weatherUtils.test.ts > formatTemperature > should format Fahrenheit temperature
     same root cause as (1)
  3. src/App.test.tsx > App > toggles temperature unit
     the toggle button renders the wrong label

FILES IMPLICATED:
  - src/utils/weatherUtils.ts   -- the convertTemperature function, Celsius->Fahrenheit branch
  - src/components/TemperatureToggle.tsx -- the button label expression

ALREADY TRIED: nothing. These are untouched.

ACCEPTANCE CRITERIA: `npx vitest --run` reports 26 passed, 0 failed. Do not edit
test files. Do not refactor anything unrelated."""


def run_tests(repo):
    r = subprocess.run(["npx", "vitest", "--run"], cwd=repo,
                       capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    m = re.search(r"^\s*Tests\s+(.+?)\(\d+\)\s*$", out, re.M)
    passed = failed = 0
    if m:
        line = m.group(1)
        if f := re.search(r"(\d+)\s+failed", line): failed = int(f.group(1))
        if p := re.search(r"(\d+)\s+passed", line): passed = int(p.group(1))
    fails = re.findall(r"FAIL\s+(\S+ > .+)", out)[:6]
    summary = f"Tests: {passed} passed, {failed} failed\n" + "\n".join(f"FAIL {x}" for x in fails)
    return passed, failed, summary[:1200]


def safe_path(repo, rel):
    p = (repo / rel.strip()).resolve()
    if not str(p).startswith(str(repo.resolve())):
        return None
    return p


def execute(repo, cmd):
    """Run one restricted verb. Returns (observation, kind)."""
    cmd = cmd.strip()
    if cmd.upper().startswith("DONE"):
        return "", "done"

    if cmd.upper().startswith("TEST"):
        p, f, s = run_tests(repo)
        return s, "test"

    if cmd.upper().startswith("LIST"):
        rel = cmd.partition("\n")[0][4:].strip() or "."
        p = safe_path(repo, rel)
        if not p or not p.exists():
            return f"No such directory: {rel}", "error"
        names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()
                       if not x.name.startswith(".") and x.name != "node_modules")
        return "\n".join(names)[:1500], "list"

    if cmd.upper().startswith("READ"):
        rel = cmd.partition("\n")[0][4:].strip()
        p = safe_path(repo, rel)
        if not p or not p.is_file():
            return f"No such file: {rel}", "error"
        return p.read_text()[:6000], "read"

    if cmd.upper().startswith("WRITE"):
        first, _, rest = cmd.partition("\n")
        rel = first[5:].strip()
        if not rel or len(rel) > 200:
            return "WRITE needs a path on the same line: WRITE <path>", "error"
        if re.search(r"\.(test|spec)\.[jt]sx?$", rel):
            return (f"Refusing to write {rel}: test files are off limits."), "error"
        p = safe_path(repo, rel)
        if not p or not p.is_file():
            return f"No such file: {rel}", "error"
        body = re.sub(r"^```[a-z]*\n|\n```\s*$", "", rest.strip(), flags=re.M)
        if not body.strip():
            return "WRITE needs the full file contents on the following lines.", "error"
        before = p.read_text()
        p.write_text(body if body.endswith("\n") else body + "\n")
        # a whole-file write that loses most of the file is the failure anchor edits
        # structurally cannot have -- record it rather than silently accepting
        shrink = 1 - (len(body) / max(len(before), 1))
        note = ""
        if shrink > 0.3:
            note = (f" WARNING: file shrank {shrink:.0%} ({len(before)} -> {len(body)} "
                    "chars). If that was unintended, restore the missing code.")
        return f"Wrote {rel} ({len(body)} chars).{note}", "edit"

    if cmd.upper().startswith("EDIT"):
        # take the path from the FIRST LINE only -- a regex over the whole command
        # silently swallowed the entire SEARCH/REPLACE body as a filename
        first, _, rest = cmd.partition("\n")
        rel = first[4:].strip()
        if not rel or len(rel) > 200 or "\n" in rel:
            return "EDIT needs a path on the same line: EDIT <path>", "error"
        # protocol forbids editing tests; enforce it rather than trusting the model
        if re.search(r"\.(test|spec)\.[jt]sx?$", rel):
            return (f"Refusing to edit {rel}: test files are off limits. "
                    "Fix the source file that makes the test fail."), "error"
        m = re.search(r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE",
                      rest, re.S)
        if not m:
            return ("EDIT needs a SEARCH/REPLACE block on the following lines.",
                    "error")
        search, repl = m.group(1), m.group(2)
        p = safe_path(repo, rel)
        if not p or not p.is_file():
            return f"No such file: {rel}", "error"
        src = p.read_text()
        if search in src:
            p.write_text(src.replace(search, repl, 1)); return f"Edited {rel}.", "edit"
        s2 = "\n".join(l.rstrip() for l in search.split("\n"))
        o2 = "\n".join(l.rstrip() for l in src.split("\n"))
        if s2 in o2:
            p.write_text(o2.replace(s2, "\n".join(l.rstrip() for l in repl.split("\n")), 1))
            return f"Edited {rel}.", "edit"
        return f"SEARCH text not found in {rel}. Read the file and copy the lines exactly.", "error"

    return ("Unrecognised command. There is no shell. Use EXACTLY one of:\n"
            "  LIST <dir>\n  READ <path>\n  TEST\n  EDIT <path> + SEARCH/REPLACE block\n"
            "  DONE\nTo run the tests, reply with just: TEST"), "error"


def run_arm(arm, opening, port, protocol):
    repo = WS / f"run_{arm}"
    if repo.exists():
        shutil.rmtree(repo)
    shutil.copytree(PRISTINE, repo, symlinks=True,
                    ignore=shutil.ignore_patterns("node_modules"))
    (repo / "node_modules").symlink_to(
        Path.home() / "Desktop/localaitest/qwen3coder/weather-workbench/node_modules")

    p0, f0, _ = run_tests(repo)
    print(f"  start: {p0} passed / {f0} failed", flush=True)

    msgs = [{"role": "system", "content": protocol},
            {"role": "user", "content": opening}]
    rec = {"arm": arm, "turns": [], "start_failed": f0}
    first_edit_turn = first_edit_ctx = None
    tool_calls_before_edit = 0
    peak_ctx = 0

    for turn in range(1, MAX_TURNS + 1):
        r = mx.chat(port, msgs, max_tokens=MAX_TOKENS, timeout=900)
        if not r.get("ok"):
            rec["abort"] = r.get("error"); print(f"  turn {turn}: ERROR", flush=True); break

        ctx = r["usage"].get("prompt_tokens") or 0
        peak_ctx = max(peak_ctx, ctx)
        reply = (r["text"] or r["reasoning"] or "").strip()
        reply = re.sub(r"^```[a-z]*\n|\n```$", "", reply).strip()

        try:
            obs, kind = execute(repo, reply)
        except Exception as ex:
            obs, kind = f"Command failed: {type(ex).__name__}. Re-read the protocol.", "error"
        if kind == "edit" and first_edit_turn is None:
            first_edit_turn, first_edit_ctx = turn, ctx
            print(f"  >> FIRST EDIT at turn {turn}, context {ctx:,} tokens", flush=True)
        if kind != "edit" and first_edit_turn is None:
            tool_calls_before_edit += 1

        rec["turns"].append({"turn": turn, "ctx": ctx, "kind": kind,
                             "cmd": reply.split("\n")[0][:80],
                             "out_tokens": r["usage"].get("completion_tokens")})
        print(f"  turn {turn:>2} ctx={ctx:>7,} {kind:<6} {reply.split(chr(10))[0][:60]}",
              flush=True)

        if kind == "done":
            break
        if ctx > CONTEXT_ABORT:
            rec["abort"] = f"context exceeded {CONTEXT_ABORT}"; break

        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": obs or "(no output)"})

    p1, f1, _ = run_tests(repo)
    rec.update({
        "first_edit_turn": first_edit_turn,
        "context_at_first_edit": first_edit_ctx,
        "tool_calls_before_first_edit": tool_calls_before_edit,
        "peak_context": peak_ctx,
        "total_turns": len(rec["turns"]),
        "end_passed": p1, "end_failed": f1,
        "solved": f1 == 0 and p1 >= 26,
    })
    print(f"  -> solved={rec['solved']} ({p1} passed / {f1} failed) "
          f"first_edit_ctx={first_edit_ctx} peak={peak_ctx:,} turns={rec['total_turns']}",
          flush=True)
    return rec


def main():
    if not PRISTINE.exists():
        sys.exit("e4_workspace/pristine missing")
    results = {"model": MODEL, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "hypothesis": "does the diff-only advantage survive multi-turn, or does whole-file become more stable (Aider)?",
               "arms": []}
    proc = mx.start_server(MODEL, PORT)
    try:
        mx.wait_ready(PORT, proc)
        print(f"weights {mx.rss_gib(proc.pid)} GiB\n", flush=True)
        for arm, protocol in (("anchor_edits", PROTOCOL_ANCHOR),
                              ("whole_file", PROTOCOL_WHOLE)):
            print(f"=== {arm} ===", flush=True)
            results["arms"].append(run_arm(arm, BRIEF, PORT, protocol))
            RESULTS.write_text(json.dumps(results, indent=2))
    finally:
        mx.stop_server(proc)
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nDONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        mx._kill_all()
