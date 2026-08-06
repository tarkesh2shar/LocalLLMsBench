#!/usr/bin/env python3
"""
Cheap pre-download screen. One HTTP request per model; no weights downloaded.

Checks the three things that decide whether a model is usable at all:

  1. Is `model_type` in mlx-lm's supported architecture list?
     (mlx-lm 0.31.3 supports 119; a miss means it will NOT load, e.g.
      North-Mini-Code-1.0 -> "ValueError: Model type cohere2_moe not supported")

  2. Does ANY layer do full attention?
     Do NOT screen on `sliding_window` alone. gpt-oss-20b has
     sliding_window=128 yet recalls a token planted 20K back, because 12 of its
     24 layers are full attention. Screening on the window alone wrongly
     rejects every interleaved-attention model (gpt-oss, Gemma, Mistral).

  3. KV bytes/token = 2 * full_attention_layers * kv_heads * head_dim * 2
     Under ~100 KiB/token is comfortable on 48 GB. 256 KiB/token (Seed-OSS)
     makes long context impractical.

Usage:
    python3 screen_config.py mlx-community/gpt-oss-20b-MXFP4-Q8 [more repos...]
    python3 screen_config.py --local        # screen everything already cached
"""

import glob
import json
import os
import sys
import urllib.request

MLX_MODELS_GLOB = os.path.expanduser(
    "~/.local/share/uv/tools/mlx-lm/lib/python*/site-packages/mlx_lm/models/*.py")


def supported_types():
    return {os.path.basename(p)[:-3] for p in glob.glob(MLX_MODELS_GLOB)}


def fetch_config(repo):
    url = f"https://huggingface.co/{repo}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def analyse(cfg, supported):
    text = cfg.get("text_config", cfg)
    mt = cfg.get("model_type")
    layers = text.get("num_hidden_layers")
    kv_heads = text.get("num_key_value_heads")
    head_dim = text.get("head_dim")

    layer_types = text.get("layer_types")
    if isinstance(layer_types, list):
        full = sum(1 for t in layer_types if "full" in str(t))
    elif text.get("full_attention_interval") and layers:
        full = layers // text["full_attention_interval"]
    else:
        full = layers  # no interleaving declared -> assume all full

    kv = None
    if all(isinstance(v, int) for v in (kv_heads, head_dim)) and full:
        kv = 2 * full * kv_heads * head_dim * 2 / 1024  # KiB/token

    return {
        "model_type": mt,
        "loadable": mt in supported,
        "sliding_window": text.get("sliding_window"),
        "layers": layers,
        "full_attention_layers": full,
        "kv_kib_per_token": round(kv, 1) if kv else None,
        "max_context": text.get("max_position_embeddings"),
        "multimodal": "vision_config" in cfg,
    }


def verdict(a):
    if not a["loadable"]:
        return "REJECT - mlx-lm cannot load this architecture"
    if a["full_attention_layers"] in (0, None):
        return "REJECT - no full-attention layers (cannot retain long context)"
    if a["kv_kib_per_token"] and a["kv_kib_per_token"] > 200:
        return "CAUTION - very expensive KV; keep context short"
    return "OK"


def main():
    supported = supported_types()
    if not supported:
        print("warning: could not find mlx-lm models dir; "
              "architecture check disabled", file=sys.stderr)

    repos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not repos:
        print(__doc__)
        return

    for repo in repos:
        try:
            a = analyse(fetch_config(repo), supported)
        except Exception as e:
            print(f"\n{repo}\n  ERROR: {type(e).__name__}: {e}")
            continue
        print(f"\n{repo}")
        print(f"  model_type          {a['model_type']}  "
              f"({'supported' if a['loadable'] else 'NOT SUPPORTED'})")
        print(f"  sliding_window      {a['sliding_window']}")
        print(f"  full-attn layers    {a['full_attention_layers']} of {a['layers']}")
        print(f"  KV per token        {a['kv_kib_per_token']} KiB")
        print(f"  max context         {a['max_context']}")
        print(f"  -> {verdict(a)}")


if __name__ == "__main__":
    main()
