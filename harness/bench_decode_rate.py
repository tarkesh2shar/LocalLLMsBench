#!/usr/bin/env python3
"""Fixed-workload decode-rate probe against an ALREADY-RUNNING server.

Isolates raw generation speed from task difficulty: one prompt, one token cap,
greedy. Because decoding is greedy the output is identical across server configs,
so `sha` doubles as proof that a speculation setting changed only the speed --
if two arms share a sha, the comparison is clean.

Used to answer two questions the graded suites cannot:
  1. does --spec-draft-n-max want tuning?  (no: 3, the default, is optimal;
     2 ties it, 4 and 6 are progressively worse)
  2. is MLX or llama.cpp faster on the same weights?  (MLX by ~7% at plain
     decode, which llama.cpp more than reverses with MTP)

    python3 bench_decode_rate.py --port 8095 --label "n-max=3 (default)"
"""

import argparse
import hashlib
import json
import time
import urllib.request

PROMPT = ("Write a complete TypeScript implementation of a generic LRU cache class "
          "with get, set, has, delete and clear, an explicit capacity, and doc comments "
          "on every public method. Then write a second version using a doubly linked "
          "list. Output only code.")


def run(port, max_tokens, effort=None):
    body = {"messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "stream": False, "temperature": 0, "top_k": 1}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    el = time.time() - t0
    msg = d["choices"][0]["message"]
    # mlx_lm.server returns NEITHER content nor reasoning_content when the
    # reasoning block is still open at the cap, so this hashes to the empty
    # string on that runtime. It is a runtime difference, not a failed run --
    # completion_tokens is still authoritative.
    text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return {"elapsed_s": round(el, 2), "completion_tokens": ct,
            "tok_per_s": round(ct / el, 2) if el else None,
            "finish": d["choices"][0].get("finish_reason"),
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--effort", default=None)
    a = ap.parse_args()

    run(a.port, 64, a.effort)          # warm-up: first call pays model load
    rec = run(a.port, a.max_tokens, a.effort)
    rec = {"label": a.label, "effort": a.effort, **rec}
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
