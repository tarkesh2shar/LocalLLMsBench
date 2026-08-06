#!/usr/bin/env python3
"""
MTP speculative decoding via llama.cpp.

mlx-lm cannot do this at all: the published MTP drafter is model_type
qwen3_5_mtp, which mlx-lm does not implement, and mlx-lm has no native MTP
pipeline -- so the MTP heads shipped inside Qwen3.5/3.6 go unused there.

llama.cpp has an independent speculative implementation, so it may also avoid the
output divergence measured through mlx-lm (3 of 5 runs differed from baseline, and
the divergence changed with draft length -- which correct greedy speculative
decoding must not do).

Baseline for comparison, already measured (same runtime, no drafter):
    Qwen3.6-27B GGUF Q4_0 : 3/5, 1608s, 15.3 tok/s, 16.37 GiB resident
"""
import json, re, sys, time
from pathlib import Path

import llama_server as ls
import mlx_server as mx
import bench_extended as bx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results-mtp.json"
GGUF = Path.home() / "models" / "gguf"
TARGET = GGUF / "Qwen3.6-27B-Q4_0.gguf"
DRAFT = GGUF / "mtp-Qwen3.6-27B-Q4_0.gguf"
PORT = 8096

ARMS = [("mtp_n3", 3), ("mtp_n5", 5)]

BASELINE = {"score": "3/5", "total_s": 1607.7, "tok_per_s": 15.3, "rss_gib": 16.37,
            "per_task_s": {"T3_runtime_bug/whole_file": 208.57,
                           "T3_runtime_bug/search_replace": 55.5,
                           "T4_implement_spec/whole_file": 319.08,
                           "T5_noop_trap/whole_file": 518.1,
                           "T5_noop_trap/search_replace": 506.46}}


def canary(port):
    r = mx.chat_stream(port, [{"role": "user", "content":
        "Count from 1 to 20. Output only the numbers separated by commas, nothing else."}],
        max_tokens=400, stall_timeout=180, hard_timeout=600)
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error")}
    text = bx.strip_reasoning(r["text"]) or bx.strip_reasoning(r["reasoning"])
    nums = re.findall(r"\d+", text)
    return {"ok": True, "missing": [i for i in range(1, 21) if str(i) not in nums],
            "tok_per_s": round((r["usage"].get("completion_tokens") or 0)
                               / max(r["elapsed"], .01), 1), "text": text[-200:]}


def main():
    for f in (TARGET, DRAFT):
        if not f.exists():
            sys.exit(f"missing {f}")
    bx.ensure_orig(); bx.restore_all()
    if len(bx.tsc_errors()) != 2:
        sys.exit("fixture not pristine")
    print("baseline tsc errors: 2", flush=True)

    results = {"target": str(TARGET), "draft": str(DRAFT),
               "no_drafter_baseline": BASELINE, "arms": []}

    for arm, n in ARMS:
        print(f"\n=== {arm} (spec-draft-n-max={n}) ===", flush=True)
        entry = {"arm": arm, "draft_n": n, "runs": []}
        proc = None
        try:
            proc = ls.start_server(TARGET, PORT, draft_gguf=DRAFT, draft_n=n,
                                   spec_type="draft-mtp")
            ls.wait_ready(PORT, proc)
            entry["rss_gib"] = mx.rss_gib(proc.pid)
            print(f"  resident {entry['rss_gib']} GiB "
                  f"(baseline {BASELINE['rss_gib']})", flush=True)
            entry["canary"] = canary(PORT)
            print(f"  canary: missing={entry['canary'].get('missing')} "
                  f"{entry['canary'].get('tok_per_s')} tok/s", flush=True)
            for task in bx.TASKS:
                for a in task["arms"]:
                    rec = bx.run_task(PORT, task, a)
                    entry["runs"].append(rec)
                    key = f"{rec['task']}/{a}"
                    b = BASELINE["per_task_s"].get(key)
                    sp = f" ({b/rec['elapsed_s']:.2f}x)" if b and rec.get("elapsed_s") else ""
                    print(f"  {key}: passed={rec['passed']} "
                          f"tok={rec.get('completion_tokens')} "
                          f"{rec.get('elapsed_s')}s{sp}", flush=True)
        except Exception as e:
            entry["fatal"] = f"{type(e).__name__}: {e}"
            print(f"  FATAL {e}", flush=True)
        finally:
            if proc:
                ls.stop_server(proc)
            bx.restore_all()
        p = sum(1 for r in entry["runs"] if r.get("passed"))
        sec = sum(r.get("elapsed_s") or 0 for r in entry["runs"])
        tok = sum(r.get("completion_tokens") or 0 for r in entry["runs"])
        entry["score"] = f"{p}/{len(entry['runs'])}"
        entry["total_s"] = round(sec, 1)
        entry["tok_per_s"] = round(tok / sec, 1) if sec else None
        if sec:
            entry["speedup_vs_baseline"] = round(BASELINE["total_s"] / sec, 2)
        print(f"  -> {entry['score']} in {sec:.0f}s ({entry['tok_per_s']} tok/s, "
              f"{entry.get('speedup_vs_baseline')}x vs no drafter)", flush=True)
        results["arms"].append(entry)
        RESULTS.write_text(json.dumps(results, indent=2))

    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nDONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        ls._kill_all()
        try: bx.restore_all()
        except Exception: pass
