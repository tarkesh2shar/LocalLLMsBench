"""
mlx_lm.server lifecycle + OpenAI-compatible client.

Handles the two things that silently break local MLX benchmarking:

1. MLX dlopen()s "libmpi.dylib" and hard-aborts (SIGABRT, before loading any
   weights) if that resolves to Anaconda's MPICH instead of Open MPI. Setting
   DYLD_LIBRARY_PATH does NOT fix it. MLX_MPI_LIBNAME does.

2. Thinking models may return the answer in `message.reasoning_content` with
   `message.content` absent entirely. A client reading only `content` sees
   nothing and appears to hang.
"""

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "results" / "server-logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _openmpi_lib():
    """Absolute path to Open MPI's libmpi.dylib, or '' if Homebrew isn't present."""
    try:
        p = subprocess.run(["brew", "--prefix", "open-mpi"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{p}/lib/libmpi.dylib" if p else ""
    except Exception:
        return ""


OPENMPI_LIB = os.environ.get("MLX_MPI_LIBNAME", _openmpi_lib())

_servers = []


def _kill_all():
    for p in list(_servers):
        stop_server(p, quiet=True)


atexit.register(_kill_all)
for _s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_s, lambda *_: (_kill_all(), sys.exit(1)))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def start_server(model, port, extra_args=None, prefill_step="512"):
    """Launch mlx_lm.server for one model.

    prefill_step: --prefill-step-size. The default of 2048 drives a large
    activation spike during long-prompt prefill; 512 measurably lowered peak
    memory AND increased prefill throughput on a 48 GB M5 Pro.
    """
    if not port_free(port):
        raise RuntimeError(f"port {port} already in use")
    logf = open(LOG_DIR / f"server_{port}.log", "w")
    cmd = ["mlx_lm.server", "--model", model, "--host", "127.0.0.1",
           "--port", str(port), "--log-level", "INFO",
           "--prefill-step-size", prefill_step] + list(extra_args or [])
    env = dict(os.environ)
    if OPENMPI_LIB:
        env["MLX_MPI_LIBNAME"] = OPENMPI_LIB
    log(f"start: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         start_new_session=True, env=env)
    p._logf, p._port = logf, port
    _servers.append(p)
    return p


def stop_server(p, quiet=False):
    if p.poll() is not None:
        if p in _servers:
            _servers.remove(p)
        return
    if not quiet:
        log("stopping server")
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


def wait_ready(port, proc, timeout=600):
    """Poll until the server answers a real completion (weights fully loaded)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            tail = (LOG_DIR / f"server_{port}.log").read_text()[-800:]
            raise RuntimeError(f"server exited (code {proc.returncode}):\n{tail}")
        try:
            if chat(port, [{"role": "user", "content": "hi"}],
                    max_tokens=1, timeout=60).get("ok"):
                log(f"ready in {time.time()-t0:.0f}s")
                return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("server did not become ready")


def chat(port, messages, max_tokens=800, timeout=1800, **params):
    """POST /v1/chat/completions.

    Passing no sampling params reproduces what an OpenAI-compatible client
    sends by default -- which means mlx_lm.server's own default of
    `--temp 0.0`, i.e. fully greedy decoding.
    """
    body = {"messages": messages, "max_tokens": max_tokens, "stream": False}
    body.update(params)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.read()[:300].decode('replace')}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    msg = data["choices"][0]["message"]
    return {
        "ok": True,
        "text": msg.get("content") or "",
        # thinking models may put everything here and omit `content` entirely
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "message_keys": list(msg.keys()),
        "finish": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage", {}),
        "elapsed": round(time.time() - t0, 2),
    }


def rss_gib(pid):
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                       capture_output=True, text=True)
    try:
        return round(int(r.stdout.strip()) * 1024 / 1024 ** 3, 2)
    except ValueError:
        return 0.0
