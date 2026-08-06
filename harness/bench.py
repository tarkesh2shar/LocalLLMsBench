#!/usr/bin/env python3
"""
Graded worker-role benchmark for local MLX models.

Question it answers: can a local model reliably complete a small, well-scoped
code edit handed to it by a planner?

Grading is objective -- the model proposes an edit, we apply it to an isolated
copy of the fixture and run `tsc`. A run passes only if the target error is gone
AND no new error appeared. Pre-existing errors in files the task did not ask
about are expected to remain and are not counted against the model.

Two edit strategies are compared, because that is a design decision and not a
curiosity:
  whole_file      -- model returns the entire corrected file
  search_replace  -- model returns one anchor-based SEARCH/REPLACE block

Usage:
    python3 bench.py                 # all models in MODELS
    python3 bench.py gpt-oss         # substring filter
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

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixture"
RESULTS = ROOT / "results" / "results.json"

# (repo, port, extra server args). Thinking is left ON where a model has it:
# Qwen3.5 scored 2/4 with thinking vs 1/4 without.
MODELS = [
    ("mlx-community/Qwen3.6-35B-A3B-4bit", 8085, []),
    ("mlx-community/gpt-oss-20b-MXFP4-Q8", 8084, []),
    ("mlx-community/Qwen3-Coder-30B-A3B-Instruct-5bit", 8081, []),
    ("mlx-community/Qwen3.5-35B-A3B-4bit", 8082, []),
    ("mlx-community/Qwen3.6-27B-4bit", 8086, []),
    ("mlx-community/granite-4.0-h-small-4bit", 8087, []),
    ("mlx-community/Seed-OSS-36B-Instruct-4bit", 8088, []),
    ("mlx-community/mistralai_Devstral-Small-2-24B-Instruct-2512-MLX-4Bit", 8083, []),
]

# large enough for a thinking model to reason AND still answer; too small a cap
# is consumed entirely by reasoning and returns empty content
MAX_TOKENS = 8000

TASKS = [
    {
        "id": "T1_shadowed_identifier",
        "file": "src/App.tsx",
        "error": "src/App.tsx(149,16): error TS6133: 'error' is declared but its value is never read.",
        "criteria": "TS6133 in App.tsx is gone and no new type errors appear.",
        # `error` is BOTH a useState variable and a .catch() parameter. The fix is
        # `.catch(error =>` -> `.catch(() =>`. Models that try to resolve "line 149"
        # instead of locating the code semantically hallucinate the line contents.
    },
    {
        "id": "T2_missing_namespace",
        "file": "src/components/SearchBar.tsx",
        "error": "src/components/SearchBar.tsx(30,20): error TS2503: Cannot find namespace 'NodeJS'.",
        "criteria": "TS2503 in SearchBar.tsx is gone and no new type errors appear.",
        # fix: NodeJS.Timeout -> ReturnType<typeof setTimeout>
    },
]

WHOLE_FILE = """You are fixing one specific TypeScript build error in a React project.

FILE: {path}
```tsx
{content}
```

BUILD ERROR TO FIX:
{error}

ACCEPTANCE CRITERIA: {criteria}

Fix ONLY this error. Do not refactor anything else. Do not change behaviour.
Output the COMPLETE corrected file inside a single ```tsx fenced code block.
Output nothing else."""

SEARCH_REPLACE = """You are fixing one specific TypeScript build error in a React project.

FILE: {path}
```tsx
{content}
```

BUILD ERROR TO FIX:
{error}

ACCEPTANCE CRITERIA: {criteria}

Fix ONLY this error. Do not refactor anything else. Do not change behaviour.
Output a SINGLE edit in exactly this format, and nothing else:

<<<<<<< SEARCH
(the exact original lines to find, copied verbatim)
=======
(the replacement lines)
>>>>>>> REPLACE"""

BASELINE = []


# ------------------------------------------------------------------ utilities

def strip_reasoning(text):
    """Remove <think> blocks and gpt-oss harmony channel markers."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)
    # gpt-oss emits reasoning inline behind harmony channel tokens
    text = re.sub(r"<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>|$)",
                  "", text, flags=re.S)
    text = re.sub(r"<\|[a-z_]+\|>", "", text)
    return text.strip()


def extract_block(text):
    m = re.findall(r"```(?:tsx|ts|typescript|jsx|javascript)?\s*\n(.*?)```", text, re.S)
    return m[0] if m else None


def apply_search_replace(original, out):
    m = re.search(r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", out, re.S)
    if not m:
        return None, "no valid SEARCH/REPLACE block"
    search, repl = m.group(1), m.group(2)
    if search in original:
        return original.replace(search, repl, 1), None
    # tolerate trailing-whitespace drift only
    s2 = "\n".join(l.rstrip() for l in search.split("\n"))
    o2 = "\n".join(l.rstrip() for l in original.split("\n"))
    if s2 in o2:
        r2 = "\n".join(l.rstrip() for l in repl.split("\n"))
        return o2.replace(s2, r2, 1), None
    return None, "SEARCH text not found in file (model did not copy source verbatim)"


def restore():
    for t in TASKS:
        shutil.copy(FIXTURE / (t["file"] + ".orig"), FIXTURE / t["file"])


def tsc():
    r = subprocess.run(["npx", "tsc", "--noEmit"], cwd=FIXTURE,
                       capture_output=True, text=True, timeout=300)
    return [l for l in r.stdout.splitlines() if "error TS" in l]


def grade(task):
    errs = tsc()
    new = [e for e in errs if e not in BASELINE]
    return {
        "passed": bool(task["error"] not in errs and not new),
        "target_error_gone": task["error"] not in errs,
        "new_errors_introduced": new,
    }


# ------------------------------------------------------------------ benchmark

def run_model(model, port, server_args):
    out = {"model": model, "runs": []}
    proc = mx.start_server(model, port, server_args)
    try:
        mx.wait_ready(port, proc)
        out["weights_gib"] = mx.rss_gib(proc.pid)
        print(f"  weights {out['weights_gib']} GiB", flush=True)

        for task in TASKS:
            src = (FIXTURE / (task["file"] + ".orig")).read_text()
            for arm, tmpl in (("whole_file", WHOLE_FILE),
                              ("search_replace", SEARCH_REPLACE)):
                restore()
                prompt = tmpl.format(path=task["file"], content=src,
                                     error=task["error"], criteria=task["criteria"])
                r = mx.chat(port, [{"role": "user", "content": prompt}],
                            max_tokens=MAX_TOKENS)
                if not r.get("ok"):
                    out["runs"].append({"task": task["id"], "arm": arm,
                                        "error": r.get("error")})
                    print(f"  {task['id']}/{arm}: REQUEST ERROR", flush=True)
                    continue

                text = strip_reasoning(r["text"]) or strip_reasoning(r["reasoning"])
                if arm == "whole_file":
                    new, err = extract_block(text), None
                    if not new:
                        err = "no fenced code block in output"
                else:
                    new, err = apply_search_replace(src, text)

                changed = None
                if new:
                    (FIXTURE / task["file"]).write_text(new)
                    g = grade(task)
                    changed = sum(1 for d in difflib.unified_diff(
                        src.splitlines(), new.splitlines(), n=0)
                        if d.startswith(("+", "-")) and not d.startswith(("+++", "---")))
                else:
                    g = {"passed": False, "reason": err}

                rec = {
                    "task": task["id"], "arm": arm, "passed": g["passed"], "grade": g,
                    "completion_tokens": r["usage"].get("completion_tokens"),
                    "elapsed_s": r["elapsed"], "finish": r["finish"],
                    "lines_changed": changed,
                    "reasoning_chars": len(r["reasoning"]),
                    "message_keys": r["message_keys"],
                    "output_preview": text[:300],
                }
                out["runs"].append(rec)
                print(f"  {task['id']}/{arm}: passed={g['passed']} "
                      f"lines={changed} tok={rec['completion_tokens']} "
                      f"{rec['elapsed_s']}s", flush=True)
                restore()
    finally:
        mx.stop_server(proc)
        restore()
    return out


def main():
    if not (FIXTURE / "node_modules").exists():
        sys.exit(f"fixture deps missing -- run: cd {FIXTURE} && npm install")

    for t in TASKS:
        orig = FIXTURE / (t["file"] + ".orig")
        if not orig.exists():
            shutil.copy(FIXTURE / t["file"], orig)

    restore()
    BASELINE[:] = tsc()
    print(f"baseline errors: {len(BASELINE)}", flush=True)
    for e in BASELINE:
        print(f"  {e}", flush=True)
    if len(BASELINE) != 2:
        sys.exit("expected exactly 2 baseline errors -- fixture is not pristine")

    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "baseline_errors": BASELINE, "models": []}

    for model, port, args in MODELS:
        if only and only.lower() not in model.lower():
            continue
        print(f"\n=== {model} ===", flush=True)
        try:
            results["models"].append(run_model(model, port, args))
        except Exception as e:
            print(f"  FATAL {type(e).__name__}: {e}", flush=True)
            results["models"].append({"model": model, "fatal": str(e)})
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=2))

    restore()
    print(f"\nDONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        mx._kill_all()
        try:
            restore()
        except Exception:
            pass
