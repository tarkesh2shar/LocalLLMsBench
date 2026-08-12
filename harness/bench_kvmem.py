"""Measure real KV cost/token with MLX's allocator, and quantify how much RSS misses.

Deliberately capped at 16k single-stream (~1.5 GiB KV) — safe. The dangerous configs
are extrapolated arithmetically, not allocated.
"""
import os, gc
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

GB = 1 << 30
MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-5bit"

def rss_gb():
    import subprocess
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) / 1048576

def clear():
    for fn in ("clear_cache",):
        if hasattr(mx, fn):
            getattr(mx, fn)()
    gc.collect()

print(f"{'stage':<28}{'mlx_active':>12}{'mlx_peak':>11}{'ps_rss':>10}{'miss':>10}")
def row(stage):
    a, p, r = mx.get_active_memory()/GB, mx.get_peak_memory()/GB, rss_gb()
    print(f"{stage:<28}{a:>11.2f}G{p:>10.2f}G{r:>9.2f}G{a-r:>9.2f}G")
    return a

row("baseline")
model, tok = load(MODEL)
mx.eval(model.parameters())
clear()
w = row("weights loaded")

ids_pool = mx.array([[123] * 16384], dtype=mx.int32)
results = []
for ctx in (2048, 4096, 8192, 16384):
    clear()
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    cache = make_prompt_cache(model)
    step = 512
    for i in range(0, ctx, step):
        out = model(ids_pool[:, i:i+step], cache=cache)
        mx.eval(out)
        del out
    mx.eval(*[c.state for c in cache if hasattr(c, "state")])
    after = mx.get_active_memory()
    kv = after - before
    peak = mx.get_peak_memory()
    results.append((ctx, kv, peak))
    print(f"  ctx={ctx:>6}  kv={kv/GB:6.3f}G  {kv/ctx/1024:7.1f} KiB/token"
          f"   peak_during_prefill={peak/GB:6.2f}G  ps_rss={rss_gb():5.2f}G")
    del cache
    clear()

print()
print("=== KV/token (from the two largest, avoids fixed overhead) ===")
(c1, k1, _), (c2, k2, _) = results[-2], results[-1]
per = (k2 - k1) / (c2 - c1)
print(f"  marginal: {per/1024:.1f} KiB/token   (doc says 96 KiB)")
print()
print("=== EXTRAPOLATED fan-out (NOT allocated) ===")
print(f"  weights: {w:.1f} GiB")
for n in (1, 2, 4, 8):
    for ctx in (8192, 32768):
        tot = w + n * ctx * per / GB
        flag = "  <-- past the 29G that crashed before" if tot > 29 else ""
        print(f"  {n} agent(s) @ {ctx//1024}k: {tot:5.1f} GiB{flag}")
