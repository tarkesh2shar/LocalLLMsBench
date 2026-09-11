#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))

import llama_server as ls
import mlx_server as mx
import bench_extended as bx

MODEL_PATH = Path.home() / "models" / "gguf" / "Dirk-Qwen3.8-27B-UD-Q4_K_XL.gguf"
PORT = 8095
RESULTS_DIR = ROOT / "results"

if not MODEL_PATH.exists():
    sys.exit(f"Missing model: {MODEL_PATH}")

print("==================================================")
print(" Dirk-Qwen3.8-27B-UD-Q4_K_XL Benchmark Suite")
print("==================================================")
sys.stdout.flush()

summary = {
    "model": "Dirk-Qwen3.8-27B-UD-Q4_K_XL",
    "gguf": str(MODEL_PATH),
    "size_gb": round(MODEL_PATH.stat().st_size / (1024**3), 2),
    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    "decode_rates": [],
    "t12_scores": {},
    "extended_scores": {},
}

def ensure_clean_fixture():
    bx.ensure_orig()
    bx.restore_all()
    base = bx.tsc_errors()
    if len(base) != 2:
        sys.exit(f"Fixture not pristine (expected 2 baseline errors, got {len(base)})")

# --- Phase 1: Launch llama-server with MTP ---
print("\n[Phase 1] Starting llama-server with MTP enabled...")
proc = ls.start_server(
    MODEL_PATH,
    PORT,
    extra_args=["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--temp", "0", "--top-k", "1"]
)
try:
    ready_s = ls.wait_ready(PORT, proc)
    summary["mtp_rss_gib"] = mx.rss_gib(proc.pid)
    print(f"Server ready in {ready_s:.1f}s | Resident RSS: {summary['mtp_rss_gib']} GiB")

    # Canary
    print("\n--- Running Canary ---")
    c = mx.chat_stream(PORT, [{"role": "user", "content": "Count from 1 to 20. Output only the numbers separated by commas, nothing else."}],
                       max_tokens=400, stall_timeout=180, hard_timeout=600)
    print(f"Canary response ok: {c.get('ok')} | tok/s: {round((c.get('usage', {}).get('completion_tokens', 0) or 0) / max(c.get('elapsed', 0.01), 0.01), 1)}")

    # Decode Rate Probe
    print("\n[Phase 2] Decode Rate Probes (MTP n-max=3)...")
    import bench_decode_rate as bdr
    bdr.run(PORT, 64) # warm-up

    # Arm 1: default template
    r1 = bdr.run(PORT, 900)
    rec1 = {"label": "Dirk MTP n-max=3 (native template)", "effort": None, **r1}
    summary["decode_rates"].append(rec1)
    print(f"  Native template: {r1['tok_per_s']} tok/s ({r1['completion_tokens']} tok in {r1['elapsed_s']}s, finish={r1['finish']})")

    # Arm 2: thinking off
    r2 = bdr.run(PORT, 900, effort="off")
    rec2 = {"label": "Dirk MTP n-max=3 (thinking off)", "effort": "off", **r2}
    summary["decode_rates"].append(rec2)
    print(f"  Thinking off:    {r2['tok_per_s']} tok/s ({r2['completion_tokens']} tok in {r2['elapsed_s']}s, finish={r2['finish']})")

    # Arm 3: effort=medium
    r3 = bdr.run(PORT, 900, effort="medium")
    rec3 = {"label": "Dirk MTP n-max=3 (effort=medium)", "effort": "medium", **r3}
    summary["decode_rates"].append(rec3)
    print(f"  Effort medium:   {r3['tok_per_s']} tok/s ({r3['completion_tokens']} tok in {r3['elapsed_s']}s, finish={r3['finish']})")

    # Phase 3: T1/T2 Code Repair
    print("\n[Phase 3] T1/T2 Code Repair (Native Dirk Sharp Template)...")
    ensure_clean_fixture()
    t12_native_out = RESULTS_DIR / "results-t12-dirk-qwen3.8-27b-mtp-native.json"
    cmd_t12 = [sys.executable, str(ROOT / "harness" / "bench_t12_external.py"),
               "--port", str(PORT), "--label", "dirk-qwen3.8-27b-mtp-native",
               "--out", str(t12_native_out)]
    subprocess.run(cmd_t12, check=True)
    if t12_native_out.exists():
        t12_data = json.loads(t12_native_out.read_text())
        summary["t12_scores"]["native"] = {
            "score": t12_data.get("score"),
            "total_tokens": sum(r.get("completion_tokens", 0) for r in t12_data.get("runs", [])),
            "total_s": round(sum(r.get("elapsed_s", 0) for r in t12_data.get("runs", [])), 1)
        }
        print(f"  -> T1/T2 Native Score: {summary['t12_scores']['native']}")

    print("\n[Phase 4] T1/T2 Code Repair (Thinking Off)...")
    ensure_clean_fixture()
    t12_nothink_out = RESULTS_DIR / "results-t12-dirk-qwen3.8-27b-mtp-nothink.json"
    cmd_t12_nothink = [sys.executable, str(ROOT / "harness" / "bench_effort.py"),
                       "--port", str(PORT), "--suite", "t12", "--effort", "off",
                       "--label", "dirk-qwen3.8-27b-mtp", "--out", str(t12_nothink_out)]
    subprocess.run(cmd_t12_nothink, check=True)
    if t12_nothink_out.exists():
        t12_data2 = json.loads(t12_nothink_out.read_text())
        summary["t12_scores"]["nothink"] = {
            "score": t12_data2.get("score"),
            "total_tokens": sum(r.get("completion_tokens", 0) for r in t12_data2.get("runs", [])),
            "total_s": round(sum(r.get("elapsed_s", 0) for r in t12_data2.get("runs", [])), 1)
        }
        print(f"  -> T1/T2 Thinking Off Score: {summary['t12_scores']['nothink']}")

    print("\n[Phase 5] Extended Suite T3/T4/T5 (Thinking Off)...")
    ensure_clean_fixture()
    ext_nothink_out = RESULTS_DIR / "results-extended-dirk-qwen3.8-27b-mtp-nothink.json"
    cmd_ext = [sys.executable, str(ROOT / "harness" / "bench_effort.py"),
               "--port", str(PORT), "--suite", "extended", "--effort", "off",
               "--label", "dirk-qwen3.8-27b-mtp", "--out", str(ext_nothink_out)]
    subprocess.run(cmd_ext, check=True)
    if ext_nothink_out.exists():
        ext_data = json.loads(ext_nothink_out.read_text())
        summary["extended_scores"]["nothink"] = {
            "score": ext_data.get("score"),
            "total_tokens": ext_data.get("total_tokens"),
            "total_s": ext_data.get("total_s")
        }
        print(f"  -> Extended Suite Score: {summary['extended_scores']['nothink']}")

finally:
    print("\nStopping MTP llama-server...")
    ls.stop_server(proc)
    time.sleep(3)

# --- Phase 6: Baseline decode rate without MTP ---
print("\n[Phase 6] Starting llama-server WITHOUT MTP for baseline decode speed...")
proc_no_mtp = ls.start_server(
    MODEL_PATH,
    PORT,
    extra_args=["--temp", "0", "--top-k", "1"]
)
try:
    ls.wait_ready(PORT, proc_no_mtp)
    bdr.run(PORT, 64)
    r_base = bdr.run(PORT, 900)
    rec_base = {"label": "Dirk GGUF UD-Q4_K_XL (no MTP baseline)", "effort": None, **r_base}
    summary["decode_rates"].append(rec_base)
    print(f"  No MTP Baseline: {r_base['tok_per_s']} tok/s ({r_base['completion_tokens']} tok in {r_base['elapsed_s']}s)")
finally:
    print("\nStopping Baseline llama-server...")
    ls.stop_server(proc_no_mtp)

summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
summary_file = RESULTS_DIR / "results-dirk-qwen3.8-27b-summary.json"
summary_file.write_text(json.dumps(summary, indent=2))
print(f"\n==================================================")
print(f" Benchmark Complete! Summary written to: {summary_file}")
print(f"==================================================")
