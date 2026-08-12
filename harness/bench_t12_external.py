#!/usr/bin/env python3
"""
T1/T2 (the two tsc-graded repair tasks) against an ALREADY-RUNNING server.

bench.py owns its own MLX server lifecycle, so a model with no MLX build cannot
be scored on T1/T2 at all -- which would leave Muse-Glimmer with a /5 that is not
comparable to the /9 every MLX candidate carries. This runs the identical loop,
prompts and grader against a port someone else owns.

T1 is the interesting one: `error` is both a useState variable and a .catch()
parameter, so a model that resolves "line 149" instead of locating the code
semantically hallucinates the line's contents. It defeated every Qwen3.5-era
configuration; gpt-oss-20b solved it by explicitly refusing to count lines.

Usage:
    python3 bench_t12_external.py --port 8095 --label muse-glimmer-30b-ud-q2_k_xl
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import bench
import mlx_server as mx
from bench import (TASKS, MAX_TOKENS, WHOLE_FILE, SEARCH_REPLACE, FIXTURE,
                   strip_reasoning, extract_block, apply_search_replace,
                   restore, grade, tsc)

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--label", default="unknown-model")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if not mx.chat(a.port, [{"role": "user", "content": "hi"}],
                   max_tokens=1, timeout=180).get("ok"):
        sys.exit(f"nothing answering on port {a.port}")

    # bench.grade() scores an error as NEW unless it is in bench.BASELINE, and
    # BASELINE is populated inside bench.main() -- which this script does not call.
    # Leaving it empty makes the OTHER pre-existing error (the one belonging to the
    # task not under test, in a file the model never touched) count as damage, so a
    # canonical correct fix grades as a failure. First run of this script scored
    # 0/4 that way with `target_error_gone: true` on three of the four runs.
    # Same family as the grader bugs already logged, opposite direction: it
    # DEFLATES the score, which is why it was visible at all.
    restore()
    bench.BASELINE[:] = tsc()
    print(f"baseline errors: {len(bench.BASELINE)}", flush=True)
    for e in bench.BASELINE:
        print(f"  {e}", flush=True)
    if len(bench.BASELINE) != 2:
        sys.exit("fixture not pristine -- expected exactly 2 baseline errors")

    out = {"model": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "T1/T2 only, run against an external server",
           "baseline_errors": list(bench.BASELINE), "runs": []}
    dest = Path(a.out) if a.out else ROOT / "results" / f"results-t12-{a.label}.json"
    for task in TASKS:
        src = (FIXTURE / (task["file"] + ".orig")).read_text()
        for arm, tmpl in (("whole_file", WHOLE_FILE),
                          ("search_replace", SEARCH_REPLACE)):
            restore()
            prompt = tmpl.format(path=task["file"], content=src,
                                 error=task["error"], criteria=task["criteria"])
            r = mx.chat(a.port, [{"role": "user", "content": prompt}],
                        max_tokens=MAX_TOKENS, timeout=2400)
            if not r.get("ok"):
                out["runs"].append({"task": task["id"], "arm": arm,
                                    "error": r.get("error")})
                print(f"  {task['id']}/{arm}: REQUEST ERROR {r.get('error')}", flush=True)
                continue

            text = strip_reasoning(r["text"]) or strip_reasoning(r["reasoning"])
            if arm == "whole_file":
                new, err = extract_block(text), None
                if not new:
                    err = "no fenced code block in output"
            else:
                # NB: bench.apply_search_replace returns (new, err); the
                # bench_extended one returns (new, err, how). This imports bench's.
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

            rec = {"task": task["id"], "arm": arm, "passed": g["passed"], "grade": g,
                   "completion_tokens": r["usage"].get("completion_tokens"),
                   "elapsed_s": r["elapsed"], "finish": r["finish"],
                   "lines_changed": changed,
                   "reasoning_chars": len(r["reasoning"]),
                   "message_keys": r["message_keys"],
                   "output_preview": text[:600],
                   "reasoning_preview": r["reasoning"][:1500]}
            out["runs"].append(rec)
            print(f"  {task['id']}/{arm}: passed={g['passed']} lines={changed} "
                  f"tok={rec['completion_tokens']} {rec['elapsed_s']}s "
                  f"finish={r['finish']}", flush=True)
            dest.write_text(json.dumps(out, indent=2))
            restore()

    p = sum(1 for r in out["runs"] if r.get("passed"))
    out["score"] = f"{p}/{len(out['runs'])}"
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out['score']}  DONE -> {dest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        restore()
