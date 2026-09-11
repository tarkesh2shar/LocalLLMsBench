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

print("==================================================")
print(" Dirk Further Testing: Effort=Low & Native Tool Calling")
print("==================================================")
sys.stdout.flush()

proc = ls.start_server(
    MODEL_PATH,
    PORT,
    extra_args=["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--temp", "0", "--top-k", "1", "--jinja"]
)

try:
    ready_s = ls.wait_ready(PORT, proc)
    print(f"Server ready in {ready_s:.1f}s on port {PORT}\n")

    # 1. T1/T2 with effort=low
    print("[Test 1] T1/T2 Code Repair with effort='low'...")
    bx.ensure_orig(); bx.restore_all()
    t12_low_out = RESULTS_DIR / "results-t12-dirk-qwen3.8-27b-mtp-effort-low.json"
    cmd1 = [sys.executable, str(ROOT / "harness" / "bench_effort.py"),
            "--port", str(PORT), "--suite", "t12", "--effort", "low",
            "--label", "dirk-qwen3.8-27b-mtp", "--out", str(t12_low_out)]
    subprocess.run(cmd1, check=True)

    # 2. Extended Suite with effort=low
    print("\n[Test 2] Extended Suite (T3/T4/T5) with effort='low'...")
    bx.ensure_orig(); bx.restore_all()
    ext_low_out = RESULTS_DIR / "results-extended-dirk-qwen3.8-27b-mtp-effort-low.json"
    cmd2 = [sys.executable, str(ROOT / "harness" / "bench_effort.py"),
            "--port", str(PORT), "--suite", "extended", "--effort", "low",
            "--label", "dirk-qwen3.8-27b-mtp", "--out", str(ext_low_out)]
    subprocess.run(cmd2, check=True)

    # 3. Native Tool Calling (control_real arm)
    print("\n[Test 3] Native Tool Calling (arm: control_real)...")
    cmd3 = [sys.executable, str(ROOT / "harness" / "bench_toolcall.py"),
            "--port", str(PORT), "--label", "dirk-qwen3.8-27b-mtp",
            "--arm", "control_real"]
    subprocess.run(cmd3, check=True)

finally:
    print("\nStopping llama-server...")
    ls.stop_server(proc)

print("\nAll further tests completed successfully!")
