#!/usr/bin/env python3
"""
Does speculative decoding preserve CORRECTNESS while improving speed?

Every timing in this repo was measured with plain autoregressive decoding. That
matters, because several conclusions -- "dense is strictly dominated", "Seed-OSS
is too slow to be practical" -- rest on speed. If speculative decoding is a free
2-3x, those conclusions need revising.

There is a catch. mlx-lm issue #846 (open as of this writing) reports that
speculative decoding with Qwen3 models SKIPS TOKENS: asked to count 1 to 20 it
emits "1, 3, 5, 6, 8, 10...". So this is deliberately run through the graded
benchmark rather than a speed script -- corrupted output fails tsc/vitest, so we
measure whether spec-dec is USABLE, not merely whether it is faster.

Target: Qwen3.6-27B-4bit. The slowest model that still scores well (7/9 in
2,208s), it ships MTP heads, and a 0.24 GB purpose-built drafter exists for it.

NOTE: mlx-lm does not implement the native MTP pipeline -- it only supports an
external --draft-model. The MTP weights are used here as such a drafter.
"""

import json
import sys
import time
from pathlib import Path

import mlx_server as mx
import bench_extended as bx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results-specdec.json"

# Qwen3.6-27B cannot do speculative decoding under mlx-lm at all:
#   - its MTP drafter is model_type qwen3_5_mtp, which mlx-lm does not implement
#   - its hybrid attention uses ArraysCache, which is not trimmable, so any
#     external drafter is rejected
#   - and its vocab is 248320 vs Qwen3-0.6B's 151936 -- not even tokenizer-compatible
# Qwen3-Coder-30B has a standard trimmable qwen3_moe cache AND matching vocab.
TARGET = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-5bit"
PORT = 8091

ARMS = [
    ("baseline", []),
    ("draft_0.6b_n3", ["--draft-model", "mlx-community/Qwen3-0.6B-4bit",
                       "--num-draft-tokens", "3"]),
    ("draft_0.6b_n5", ["--draft-model", "mlx-community/Qwen3-0.6B-4bit",
                       "--num-draft-tokens", "5"]),
]

# a deterministic correctness canary that makes token-skipping obvious, since
# that is the exact symptom reported in issue #846
CANARY_PROMPT = ("Count from 1 to 20. Output only the numbers separated by commas, "
                 "nothing else.")
CANARY_EXPECT = ", ".join(str(i) for i in range(1, 21))


def canary(port):
    r = mx.chat_stream(port, [{"role": "user", "content": CANARY_PROMPT}],
                       max_tokens=200, stall_timeout=180, hard_timeout=600)
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error")}
    text = bx.strip_reasoning(r["text"]) or bx.strip_reasoning(r["reasoning"])
    # extract every integer rather than splitting on commas: a preamble
    # ("Here are the numbers: 1, 2...") or a trailing period ("...20.") makes a
    # comma-split drop the first and last values and look like token skipping
    import re
    nums = re.findall(r"\d+", text)
    missing = [i for i in range(1, 21) if str(i) not in nums]
    return {"ok": True, "exact": CANARY_EXPECT in text.replace(" ", " ").strip(),
            "numbers_returned": len(nums), "missing": missing,
            "token_skipping": bool(missing),
            "text": text[:200], "tok_per_s": round(
                (r["usage"].get("completion_tokens") or 0) / max(r["elapsed"], .01), 1)}


def main():
    if not (bx.FIXTURE / "node_modules").exists():
        sys.exit("fixture deps missing")
    bx.ensure_orig(); bx.restore_all()
    bx.BASELINE = bx.tsc_errors() if hasattr(bx, "BASELINE") else None
    base = bx.tsc_errors()
    print(f"baseline tsc errors: {len(base)}", flush=True)
    if len(base) != 2:
        sys.exit("fixture not pristine")

    results = {"target": TARGET, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "mlx_lm_issue_846": "speculative decoding reportedly skips tokens "
                                   "with Qwen3 models; canary checks for it",
               "arms": []}

    for arm, args in ARMS:
        print(f"\n=== {arm} ===", flush=True)
        entry = {"arm": arm, "server_args": args, "runs": []}
        proc = None
        try:
            proc = mx.start_server(TARGET, PORT, args)
            mx.wait_ready(PORT, proc, timeout=900)
            entry["weights_gib"] = mx.rss_gib(proc.pid)
            print(f"  weights {entry['weights_gib']} GiB", flush=True)

            c = canary(PORT)
            entry["canary"] = c
            print(f"  canary: token_skipping={c.get('token_skipping')} "
                  f"missing={c.get('missing')} {c.get('tok_per_s')} tok/s", flush=True)
            if c.get("token_skipping"):
                print("  !! reproduces mlx-lm #846 -- output is corrupted", flush=True)

            for task in bx.TASKS:
                for a in task["arms"]:
                    rec = bx.run_task(PORT, task, a)
                    entry["runs"].append(rec)
                    print(f"  {rec['task']}/{a}: passed={rec['passed']} "
                          f"tok={rec.get('completion_tokens')} "
                          f"{rec.get('elapsed_s')}s", flush=True)
        except Exception as e:
            entry["fatal"] = f"{type(e).__name__}: {e}"
            print(f"  FATAL {e}", flush=True)
        finally:
            if proc:
                mx.stop_server(proc)
            bx.restore_all()
        p = sum(1 for r in entry["runs"] if r.get("passed"))
        sec = sum(r.get("elapsed_s") or 0 for r in entry["runs"])
        tok = sum(r.get("completion_tokens") or 0 for r in entry["runs"])
        entry["score"] = f"{p}/{len(entry['runs'])}"
        entry["total_s"] = round(sec, 1)
        entry["total_tokens"] = tok
        entry["tok_per_s"] = round(tok / sec, 1) if sec else None
        print(f"  -> {entry['score']} in {sec:.0f}s ({entry['tok_per_s']} tok/s)",
              flush=True)
        results["arms"].append(entry)
        RESULTS.write_text(json.dumps(results, indent=2))

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nDONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        mx._kill_all()
        try:
            bx.restore_all()
        except Exception:
            pass
