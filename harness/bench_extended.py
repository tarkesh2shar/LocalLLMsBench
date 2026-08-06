#!/usr/bin/env python3
"""
Extended benchmark: task kinds beyond "fix a type error".

bench.py covers two `tsc`-graded repair tasks. This adds three kinds that probe
different failure modes, all still objectively graded:

  T3  runtime bug     a subtle arithmetic bug breaks an existing vitest test.
                      Graded by running vitest. Tests behavioural repair from a
                      test failure, not type-checker output.

  T4  implement       a stub function plus its spec test. The model must write
                      the implementation. Graded by vitest. Tests GENERATION
                      against acceptance criteria -- the actual worker role.

  T5  no-op trap      a clean file and a FABRICATED error message. The correct
                      behaviour is to change nothing. Nearly every failure in
                      the first benchmark involved a model inventing facts, and
                      nothing in that suite caught it directly. This does.

Built for unattended runs:
  - streaming with STALL detection, so a hung model aborts in ~5 min rather
    than blocking on a 30-minute HTTP timeout
  - health probe before each task; one automatic server restart if it died
  - partial results flushed after every single run

Usage:
    python3 bench_extended.py              # all models
    python3 bench_extended.py gpt-oss      # substring filter
"""

import difflib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mlx_server as mx
from bench import (MODELS, MAX_TOKENS, WHOLE_FILE, SEARCH_REPLACE,
                   strip_reasoning, extract_block, apply_search_replace)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixture"
RESULTS = ROOT / "results" / "results-extended.json"

STALL_TIMEOUT = 300      # seconds of silence before declaring a model hung
HARD_TIMEOUT = 1500      # total ceiling per response

UTILS = "src/utils/weatherUtils.ts"
WIND = "src/utils/windDirection.ts"

# ---------------------------------------------------------------- task defs

T3_PROMPT = """A unit test is failing in a React + TypeScript project.

FILE: {path}
```ts
{content}
```

FAILING TEST OUTPUT:
```
FAIL src/utils/weatherUtils.test.ts > Weather Utilities > convertTemperature > should convert Celsius to Fahrenheit
AssertionError: expected 30 to be 32
 ❯ src/utils/weatherUtils.test.ts:41:47
     39|   describe('convertTemperature', () => {{
     40|     it('should convert Celsius to Fahrenheit', () => {{
     41|       expect(convertTemperature(0, 'fahrenheit')).toBe(32);
```

ACCEPTANCE CRITERIA: the failing test passes, and no other test breaks.

Fix ONLY this bug. Do not refactor anything else.
{format_instructions}"""

T4_PROMPT = """Implement a function in a React + TypeScript project.

FILE: {path}
```ts
{content}
```

THIS SPEC TEST MUST PASS:
```ts
{spec}
```

ACCEPTANCE CRITERIA: every assertion in the spec test above passes.

Replace the `throw new Error('not implemented')` with a working implementation.
Do not modify the test file. Do not add dependencies.
{format_instructions}"""

T5_PROMPT = """You are fixing a TypeScript build error in a React project.

FILE: {path}
```ts
{content}
```

BUILD ERROR TO FIX:
{error}

ACCEPTANCE CRITERIA: the reported error is resolved and no new errors appear.

Fix ONLY this error. Do not refactor anything else.
If the reported error does not actually exist in the file shown, make NO change
and reply with exactly: NO CHANGE NEEDED
{format_instructions}"""

FMT_WHOLE = ("\nOutput the COMPLETE corrected file inside a single ```ts fenced "
             "code block. Output nothing else.")
FMT_SR = """
Output a SINGLE edit in exactly this format, and nothing else:

<<<<<<< SEARCH
(the exact original lines to find, copied verbatim)
=======
(the replacement lines)
>>>>>>> REPLACE"""


def vitest(test_file):
    """Run one vitest file. Returns (passed, failed, raw_tail)."""
    r = subprocess.run(["npx", "vitest", "--run", test_file],
                       cwd=FIXTURE, capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    # vitest summary varies:
    #   "Tests  10 passed (10)"            all pass
    #   "Tests  2 failed | 8 passed (10)"  mixed
    #   "Tests  4 failed (4)"              all fail -- no "passed" clause at all
    m = re.search(r"^\s*Tests\s+(.+?)\(\d+\)\s*$", out, re.M)
    if m:
        line = m.group(1)
        failed = int(f.group(1)) if (f := re.search(r"(\d+)\s+failed", line)) else 0
        passed = int(p.group(1)) if (p := re.search(r"(\d+)\s+passed", line)) else 0
        return passed, failed, out[-400:]
    if "no test files found" in out.lower():
        return 0, -1, out[-400:]
    return 0, -1, out[-400:]


def tsc_errors():
    r = subprocess.run(["npx", "tsc", "--noEmit"], cwd=FIXTURE,
                       capture_output=True, text=True, timeout=300)
    return [l for l in r.stdout.splitlines() if "error TS" in l]


TASKS = [
    {
        "id": "T3_runtime_bug",
        "kind": "vitest",
        "file": UTILS,
        "test_file": "src/utils/weatherUtils.test.ts",
        "prompt": T3_PROMPT,
        # inject the bug: +32 -> +30 in the Celsius->Fahrenheit conversion
        "setup": lambda src: src.replace("return (celsius * 9) / 5 + 32;",
                                         "return (celsius * 9) / 5 + 30;"),
        "arms": ["whole_file", "search_replace"],
    },
    {
        "id": "T4_implement_spec",
        "kind": "vitest",
        "file": WIND,
        "test_file": "src/utils/windDirection.test.ts",
        "prompt": T4_PROMPT,
        "setup": None,
        "arms": ["whole_file"],   # writing a new body is not an anchor edit
    },
    {
        "id": "T5_noop_trap",
        "kind": "noop",
        "file": UTILS,
        "prompt": T5_PROMPT,
        # this error does not exist anywhere in the file
        "error": "src/utils/weatherUtils.ts(42,10): error TS2322: Type 'number' is not assignable to type 'string'.",
        "setup": None,
        "arms": ["whole_file", "search_replace"],
    },
]


# ---------------------------------------------------------------- fixture mgmt

def ensure_orig():
    for f in (UTILS, WIND):
        o = FIXTURE / (f + ".orig")
        if not o.exists():
            shutil.copy(FIXTURE / f, o)


def restore_all():
    for f in (UTILS, WIND):
        o = FIXTURE / (f + ".orig")
        if o.exists():
            shutil.copy(o, FIXTURE / f)


def prepare(task):
    """Write the task's starting state; return the source the model will see."""
    restore_all()
    src = (FIXTURE / (task["file"] + ".orig")).read_text()
    if task["setup"]:
        src = task["setup"](src)
        (FIXTURE / task["file"]).write_text(src)
    return src


# ---------------------------------------------------------------- grading

def grade(task, new_src, applied, err):
    if not applied:
        # T5 is the exception: declining to edit IS the correct answer
        if task["kind"] == "noop":
            return {"passed": True, "declined": True, "detail": err}
        return {"passed": False, "reason": err}

    if task["kind"] == "noop":
        orig = (FIXTURE / (task["file"] + ".orig")).read_text()
        unchanged = new_src.strip() == orig.strip()
        errs = tsc_errors()
        return {"passed": bool(unchanged and len(errs) <= 2),
                "left_file_unchanged": unchanged,
                "tsc_errors_after": len(errs),
                "detail": "invented a fix for a non-existent error" if not unchanged else "correctly made no change"}

    if task["kind"] == "vitest":
        passed, failed, tail = vitest(task["test_file"])
        errs = tsc_errors()
        return {"passed": bool(failed == 0 and passed > 0 and len(errs) <= 2),
                "tests_passed": passed, "tests_failed": failed,
                "tsc_errors_after": len(errs), "vitest_tail": tail}

    return {"passed": False, "reason": "unknown task kind"}


# ---------------------------------------------------------------- run

def run_task(port, task, arm):
    src = prepare(task)
    fmt = FMT_WHOLE if arm == "whole_file" else FMT_SR
    kwargs = {"path": task["file"], "content": src, "format_instructions": fmt}
    if task["id"] == "T4_implement_spec":
        kwargs["spec"] = (FIXTURE / task["test_file"]).read_text()
    if task["kind"] == "noop":
        kwargs["error"] = task["error"]
    prompt = task["prompt"].format(**kwargs)

    r = mx.chat_stream(port, [{"role": "user", "content": prompt}],
                       max_tokens=MAX_TOKENS, stall_timeout=STALL_TIMEOUT,
                       hard_timeout=HARD_TIMEOUT)
    if not r.get("ok"):
        return {"task": task["id"], "arm": arm, "passed": False,
                "error": r.get("error"), "stalled": r.get("stalled", False),
                "timed_out": r.get("timed_out", False),
                "elapsed_s": r.get("elapsed")}

    text = strip_reasoning(r["text"]) or strip_reasoning(r["reasoning"])
    declined = bool(re.search(r"\bNO CHANGE NEEDED\b", text, re.I))

    if declined:
        new, err = None, "model replied NO CHANGE NEEDED"
    elif arm == "whole_file":
        new = extract_block(text)
        err = None if new else "no fenced code block in output"
    else:
        new, err = apply_search_replace(src, text)

    changed = None
    if new:
        (FIXTURE / task["file"]).write_text(new)
        changed = sum(1 for d in difflib.unified_diff(
            src.splitlines(), new.splitlines(), n=0)
            if d.startswith(("+", "-")) and not d.startswith(("+++", "---")))

    g = grade(task, new, bool(new), err)
    restore_all()
    return {
        "task": task["id"], "arm": arm, "passed": g["passed"], "grade": g,
        "declined_explicitly": declined,
        "completion_tokens": r["usage"].get("completion_tokens"),
        "elapsed_s": r["elapsed"], "finish": r["finish"],
        "lines_changed": changed,
        "reasoning_chars": len(r["reasoning"]),
        "output_preview": text[:250],
    }


def run_model(model, port, server_args):
    out = {"model": model, "runs": []}
    proc = mx.start_server(model, port, server_args)
    restarts = 0
    try:
        mx.wait_ready(port, proc)
        out["weights_gib"] = mx.rss_gib(proc.pid)
        print(f"  weights {out['weights_gib']} GiB", flush=True)

        for task in TASKS:
            for arm in task["arms"]:
                # health poll: if the server died, restart it once
                if proc.poll() is not None or not mx.healthy(port):
                    if restarts >= 1:
                        print("  server unhealthy and already restarted; "
                              "abandoning model", flush=True)
                        out["aborted"] = "server unhealthy after restart"
                        return out
                    print("  server unhealthy -> restarting", flush=True)
                    mx.stop_server(proc)
                    time.sleep(10)
                    proc = mx.start_server(model, port, server_args)
                    mx.wait_ready(port, proc)
                    restarts += 1

                t0 = time.time()
                rec = run_task(port, task, arm)
                out["runs"].append(rec)
                flag = ""
                if rec.get("stalled"):
                    flag = "  [STALLED]"
                elif rec.get("timed_out"):
                    flag = "  [TIMEOUT]"
                print(f"  {rec['task']}/{arm}: passed={rec['passed']} "
                      f"lines={rec.get('lines_changed')} "
                      f"tok={rec.get('completion_tokens')} "
                      f"{rec.get('elapsed_s')}s{flag}", flush=True)
                _flush(rec, model)
    finally:
        mx.stop_server(proc)
        restore_all()
    out["restarts"] = restarts
    return out


_partial = {"models": []}


def _flush(rec, model):
    """Persist after every run so an interrupted session loses nothing."""
    for m in _partial["models"]:
        if m["model"] == model:
            m["runs"].append(rec)
            break
    else:
        _partial["models"].append({"model": model, "runs": [rec]})
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(_partial, indent=2))


def main():
    if not (FIXTURE / "node_modules").exists():
        sys.exit(f"fixture deps missing -- run: cd {FIXTURE} && npm install")
    ensure_orig()
    restore_all()

    base = tsc_errors()
    print(f"baseline tsc errors: {len(base)}", flush=True)
    if len(base) != 2:
        sys.exit("fixture not pristine (expected 2 baseline errors)")

    only = sys.argv[1] if len(sys.argv) > 1 else None
    _partial["started"] = time.strftime("%Y-%m-%d %H:%M:%S")

    for model, port, args in MODELS:
        if only and only.lower() not in model.lower():
            continue
        print(f"\n=== {model} ===", flush=True)
        try:
            run_model(model, port, args)
        except Exception as e:
            print(f"  FATAL {type(e).__name__}: {e}", flush=True)
            _partial["models"].append({"model": model, "fatal": str(e)})
            RESULTS.write_text(json.dumps(_partial, indent=2))

    _partial["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RESULTS.write_text(json.dumps(_partial, indent=2))
    print(f"\nDONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        mx._kill_all()
        try:
            restore_all()
        except Exception:
            pass
