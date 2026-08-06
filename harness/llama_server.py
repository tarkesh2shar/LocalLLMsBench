"""
llama.cpp server lifecycle, exposing the same interface as mlx_server.

Why this exists: some models have no MLX build at all. Bonsai (Qwen3.6-27B
compressed to 1-bit / ternary) ships only as GGUF, and stock llama.cpp already
supports its quant types:

    40 or Q1_0  : 1.125 bpw quantization
    41 or Q2_0  : 2.25 bpw quantization (group 64)

Both llama-server and mlx_lm.server speak the OpenAI chat-completions API, so
mlx_server.chat / chat_stream work unchanged against either.

IMPORTANT -- runtime is a confound. Comparing an MLX 4-bit model against a
llama.cpp 1-bit model changes two variables at once. Always run a same-runtime
control (Qwen3.6-27B Q4_0 GGUF) before attributing a difference to quantization.
"""

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from mlx_server import LOG_DIR, chat, log, port_free, rss_gib  # reuse the client

_servers = []


def _kill_all():
    for p in list(_servers):
        stop_server(p, quiet=True)


atexit.register(_kill_all)
for _s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_s, lambda *_: (_kill_all(), sys.exit(1)))


def start_server(gguf_path, port, extra_args=None, ctx=16384, draft_gguf=None,
                 n_gpu_layers=-1):
    """Launch llama-server on a GGUF file.

    n_gpu_layers=-1 offloads everything to Metal.
    draft_gguf enables llama.cpp speculative decoding -- a different
    implementation from mlx-lm's, which has an open token-skipping bug with
    Qwen3 models (ml-explore/mlx-lm#846).
    """
    if not port_free(port):
        raise RuntimeError(f"port {port} already in use")
    gguf_path = str(gguf_path)
    if not Path(gguf_path).exists():
        raise FileNotFoundError(gguf_path)

    logf = open(LOG_DIR / f"llama_{port}.log", "w")
    cmd = ["llama-server", "-m", gguf_path, "--host", "127.0.0.1",
           "--port", str(port), "-c", str(ctx), "-ngl", str(n_gpu_layers),
           "--no-webui"]
    if draft_gguf:
        cmd += ["-md", str(draft_gguf), "--draft-max", "3", "--draft-min", "1"]
    cmd += list(extra_args or [])

    log(f"start: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True)
    p._logf, p._port = logf, port
    _servers.append(p)
    return p


def stop_server(p, quiet=False):
    if p.poll() is not None:
        if p in _servers:
            _servers.remove(p)
        return
    if not quiet:
        log("stopping llama-server")
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        p.terminate()
    try:
        p.wait(timeout=25)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            p.kill()
        p.wait(timeout=15)
    try:
        p._logf.close()
    except Exception:
        pass
    if p in _servers:
        _servers.remove(p)


def wait_ready(port, proc, timeout=900):
    """Poll until the model is loaded and answering.

    GGUF load from cold cache is slower than MLX, so the default timeout is
    generous.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            tail = (LOG_DIR / f"llama_{port}.log").read_text()[-1200:]
            raise RuntimeError(f"llama-server exited ({proc.returncode}):\n{tail}")
        try:
            if chat(port, [{"role": "user", "content": "hi"}],
                    max_tokens=1, timeout=60).get("ok"):
                log(f"ready in {time.time()-t0:.0f}s")
                return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("llama-server did not become ready")
