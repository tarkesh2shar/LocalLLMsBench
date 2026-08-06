#!/usr/bin/env python3
"""
llama.cpp arm: isolate RUNTIME from QUANTIZATION.

Bonsai-27B is Qwen3.6-27B compressed end-to-end to 1-bit / ternary. Comparing it
directly against our MLX 4-bit result would change two variables at once
(quantization AND runtime), so a same-runtime control is required:

  Qwen3.6-27B MLX 4-bit   vs  Qwen3.6-27B GGUF Q4_0   -> isolates RUNTIME
  Qwen3.6-27B GGUF Q4_0   vs  Bonsai-27B Q1_0         -> isolates QUANTIZATION
  Bonsai Q1_0             vs  Ternary-Bonsai Q2_0     -> compression scheme

Same five graded tasks throughout. Correctness is comparable across runtimes;
raw timings are not strictly comparable, so treat those as indicative.

MLX reference (already measured, plain autoregressive decoding):
    Qwen3.6-27B-4bit : 3/5, 1462s, 14.8 tok/s, 14.6 GiB resident
"""

import json
import sys
import time
from pathlib import Path

import llama_server as ls
import mlx_server as mx
import bench_extended as bx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results-llamacpp.json"
GGUF = Path.home() / "models" / "gguf"
BONSAI_Q1 = next(iter((Path.home() / ".cache/huggingface/hub").glob(
    "models--prism-ml--Bonsai-27B-gguf/snapshots/*/Bonsai-27B-Q1_0.gguf")), None)

PORT = 8095

MODELS = [
    # (label, gguf path, extra llama-server args)
    ("qwen3.6-27b-gguf-q4_0", GGUF / "Qwen3.6-27B-Q4_0.gguf", []),
    ("bonsai-27b-1bit-q1_0", BONSAI_Q1, []),
    ("ternary-bonsai-27b-q2_0", GGUF / "Ternary-Bonsai-27B-Q2_0.gguf", []),
]

CANARY_PROMPT = ("Count from 1 to 20. Output only the numbers separated by commas, "
                 "nothing else.")


def canary(port):
    import re
    r = mx.chat_stream(port, [{"role": "user", "content": CANARY_PROMPT}],
                       max_tokens=200, stall_timeout=180, hard_timeout=600)
    if not r.get("ok"):
        return {"ok": False, "error": r.get("error")}
    text = bx.strip_reasoning(r["text"]) or bx.strip_reasoning(r["reasoning"])
    nums = re.findall(r"\d+", text)
    return {"ok": True, "missing": [i for i in range(1, 21) if str(i) not in nums],
            "text": text[:200],
            "tok_per_s": round((r["usage"].get("completion_tokens") or 0)
                               / max(r["elapsed"], .01), 1)}


def main():
    if not (bx.FIXTURE / "node_modules").exists():
        sys.exit("fixture deps missing")
    bx.ensure_orig(); bx.restore_all()
    base = bx.tsc_errors()
    print(f"baseline tsc errors: {len(base)}", flush=True)
    if len(base) != 2:
        sys.exit("fixture not pristine")

    only = sys.argv[1] if len(sys.argv) > 1 else None
    # MERGE with any existing results rather than overwriting: this script is
    # normally invoked once per model (to keep one model resident at a time), and
    # a plain overwrite silently destroys the previous arm's data.
    results = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "mlx_reference": {"model": "Qwen3.6-27B-4bit (MLX)",
                                 "score": "3/5", "total_s": 1462, "tok_per_s": 14.8},
               "models": []}
    if RESULTS.exists():
        try:
            prev = json.loads(RESULTS.read_text())
            keep = [m for m in prev.get("models", [])
                    if m.get("runs") and (not only or only.lower() not in
                                          m.get("label", "").lower())]
            results["models"] = keep
            print(f"merging with {len(keep)} existing arm(s)", flush=True)
        except Exception:
            pass

    for label, path, args in MODELS:
        if only and only.lower() not in label.lower():
            continue
        if path is None or not Path(path).exists():
            print(f"\n=== {label} === SKIP (missing {path})", flush=True)
            results["models"].append({"label": label, "skipped": str(path)})
            continue

        print(f"\n=== {label} ===", flush=True)
        entry = {"label": label, "gguf": str(path),
                 "size_gb": round(Path(path).stat().st_size / 1024**3, 2), "runs": []}
        proc = None
        try:
            proc = ls.start_server(path, PORT, args)
            ls.wait_ready(PORT, proc)
            entry["rss_gib"] = mx.rss_gib(proc.pid)
            print(f"  {entry['size_gb']} GB on disk, {entry['rss_gib']} GiB resident",
                  flush=True)
            c = canary(PORT)
            entry["canary"] = c
            print(f"  canary: missing={c.get('missing')} {c.get('tok_per_s')} tok/s",
                  flush=True)

            for task in bx.TASKS:
                for arm in task["arms"]:
                    rec = bx.run_task(PORT, task, arm)
                    entry["runs"].append(rec)
                    print(f"  {rec['task']}/{arm}: passed={rec['passed']} "
                          f"lines={rec.get('lines_changed')} "
                          f"tok={rec.get('completion_tokens')} "
                          f"{rec.get('elapsed_s')}s", flush=True)
                    RESULTS.write_text(json.dumps(results | {"models": results["models"] + [entry]}, indent=2))
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
        print(f"  -> {entry['score']} in {sec:.0f}s ({entry['tok_per_s']} tok/s)",
              flush=True)
        results["models"].append(entry)
        RESULTS.write_text(json.dumps(results, indent=2))

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nDONE -> {RESULTS}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        ls._kill_all()
        try:
            bx.restore_all()
        except Exception:
            pass
