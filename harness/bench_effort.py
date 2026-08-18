#!/usr/bin/env python3
"""Reasoning-budget arms for T1/T2 and T3/T4/T5, against an ALREADY-RUNNING server.

Every earlier Qwen3.8 number in this repo was measured at the model's DEFAULT
reasoning budget, and that default is not neutral: the Qwen3.8 chat template sets
`reasoning_effort` to 'xhigh' unless the caller overrides it. Nothing in bench.py
overrides it -- it strips <think> blocks after the fact -- so the shipped 5/5 in
229s was max-thinking, and the token counts were roughly 5x what the same answers
actually need.

This runs the identical prompts and graders with `chat_template_kwargs` injected
into every request, so the budget is the only variable.

    python3 bench_effort.py --port 8095 --suite t12 --effort off
    python3 bench_effort.py --port 8095 --suite extended --effort low

--effort off  sends {"enable_thinking": false}; low/medium/xhigh send
{"reasoning_effort": ...}. `xhigh` is the model default and is there to be
explicit, not because it changes anything.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx_server as mx

ROOT = Path(__file__).resolve().parent.parent
KWARGS = {"off": {"enable_thinking": False},
          "low": {"reasoning_effort": "low"},
          "medium": {"reasoning_effort": "medium"},
          "xhigh": {"reasoning_effort": "xhigh"}}


def patch(kwargs):
    """Inject chat_template_kwargs into every request the harness makes.

    Patched on the module rather than threaded through call sites so the task
    loops stay byte-identical to the ones that produced the committed results.
    """
    for name in ("chat", "chat_stream"):
        orig = getattr(mx, name)
        def wrapper(port, messages, _o=orig, _k=kwargs, **kw):
            kw["chat_template_kwargs"] = _k
            return _o(port, messages, **kw)
        setattr(mx, name, wrapper)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--suite", choices=["t12", "extended"], required=True)
    ap.add_argument("--effort", choices=list(KWARGS), required=True)
    ap.add_argument("--label", default="qwen3.8-27b-mtp")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    patch(KWARGS[a.effort])
    tag = "nothink" if a.effort == "off" else f"effort-{a.effort}"
    dest = Path(a.out) if a.out else \
        ROOT / "results" / f"results-{a.suite}-{a.label}-{tag}.json"

    if a.suite == "t12":
        import bench_t12_external as ext
        sys.argv = ["x", "--port", str(a.port), "--label", f"{a.label}-{tag}",
                    "--out", str(dest)]
        try:
            ext.main()
        finally:
            ext.restore()
        return

    import bench_extended as bx
    bx.ensure_orig(); bx.restore_all()
    base = bx.tsc_errors()
    print(f"baseline tsc errors: {len(base)}", flush=True)
    if len(base) != 2:
        sys.exit("fixture not pristine (expected 2 baseline errors)")

    out = {"model": f"{a.label}-{tag}", "kwargs": KWARGS[a.effort],
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": []}
    try:
        for task in bx.TASKS:
            for arm in task["arms"]:
                rec = bx.run_task(a.port, task, arm)
                out["runs"].append(rec)
                print(f"  {rec['task']}/{arm}: passed={rec['passed']} "
                      f"lines={rec.get('lines_changed')} "
                      f"tok={rec.get('completion_tokens')} "
                      f"{rec.get('elapsed_s')}s", flush=True)
                dest.write_text(json.dumps(out, indent=2))
    finally:
        bx.restore_all()

    p = sum(1 for r in out["runs"] if r.get("passed"))
    out["score"] = f"{p}/{len(out['runs'])}"
    out["total_s"] = round(sum(r.get("elapsed_s") or 0 for r in out["runs"]), 1)
    out["total_tokens"] = sum(r.get("completion_tokens") or 0 for r in out["runs"])
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {out['score']}  {out['total_s']}s  {out['total_tokens']} tok "
          f"-> {dest}", flush=True)


if __name__ == "__main__":
    main()
