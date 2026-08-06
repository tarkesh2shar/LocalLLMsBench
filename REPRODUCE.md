# Reproducing the benchmark

Everything here was run on macOS (Apple Silicon) with `mlx-lm` 0.31.3.

## 1. Prerequisites

```bash
brew install open-mpi
uv tool install mlx-lm
```

`open-mpi` matters more than it looks: MLX `dlopen()`s `libmpi.dylib` and aborts if
it resolves to Anaconda's MPICH. See "Troubleshooting" below.

Node is needed for the grader (`npx tsc`).

## 2. Install fixture dependencies

The fixture is a React + TypeScript app carrying two real build errors. The grader
runs `tsc` against it.

```bash
cd fixture && npm install
```

Confirm it reproduces exactly two errors — the benchmark refuses to run otherwise:

```bash
cd fixture && npx tsc --noEmit
```

Expected:

```
src/App.tsx(149,16): error TS6133: 'error' is declared but its value is never read.
src/components/SearchBar.tsx(30,20): error TS2503: Cannot find namespace 'NodeJS'.
```

## 3. Screen a model before downloading it

One HTTP request, no weights fetched. Rejects architectures `mlx-lm` cannot load and
flags expensive KV.

```bash
python3 harness/screen_config.py mlx-community/gpt-oss-20b-MXFP4-Q8
```

## 4. Download models

```bash
python3 -c "from huggingface_hub import snapshot_download as d; d(repo_id='mlx-community/gpt-oss-20b-MXFP4-Q8')"
```

`snapshot_download` resumes partial downloads, so an interrupted fetch continues
rather than restarting.

## 5. Optional: arm the memory watchdog

Recommended for anything above ~40K context. It polls every 3s and kills only the
model server — never the harness — so a run records the failure and continues.

```bash
./harness/memory_watchdog.sh &
```

Thresholds are at the top of the script (`MIN_AVAIL_GIB`, `MAX_SWAP_GIB`,
`MAX_RSS_GIB`). The default 3 GiB floor suits a 48 GB machine; raise it on 16–32 GB.

## 6. Run

There are two benchmarks. **Run both** — the second is what reordered the results.

### Round 1: the two `tsc` repair tasks (T1, T2)

```bash
python3 harness/bench.py
```

One model at a time, each started and stopped by the harness. Filter by substring:

```bash
python3 harness/bench.py gpt-oss
```

Results land in `results/results.json`; server logs in `results/server-logs/`.

### Round 2: runtime bug, implement-from-spec, no-op trap (T3–T5)

```bash
python3 harness/bench_extended.py
```

Same model list, three more task kinds, graded by `vitest` as well as `tsc`. Filter by
model, by task, or both:

```bash
python3 harness/bench_extended.py --task=T5
python3 harness/bench_extended.py gpt-oss --task=T3
```

Results land in `results/results-extended.json`.

This one is built for unattended runs and differs from `bench.py` in three ways:

- **streaming with stall detection.** A hung model aborts after `STALL_TIMEOUT` (300s of
  silence between tokens) instead of blocking on a 30-minute HTTP timeout. Runs that
  abort are logged `[STALLED]` or `[TIMEOUT]`, so "slow" and "hung" stay distinguishable.
- **health probe + one automatic restart** per model. A model whose server dies twice is
  abandoned rather than stalling the queue.
- **results flushed after every single run**, so an interrupted session loses nothing.

`STALL_TIMEOUT` must exceed worst-case *prefill*, during which no tokens are emitted at
all. 300s is comfortable for a 20K prompt on a slow dense model; lower it and you will
kill healthy runs.

## 7. Summarise

```bash
python3 - <<'EOF'
import json
d = json.load(open("results/results.json"))
for m in d["models"]:
    if "runs" not in m: continue
    p = sum(1 for r in m["runs"] if r.get("passed"))
    tok = sum(r.get("completion_tokens") or 0 for r in m["runs"])
    sec = sum(r.get("elapsed_s") or 0 for r in m["runs"])
    print(f'{m["model"].split("/")[-1]:<44}{p}/4  {tok:>6} tok  {sec:>6.0f}s')
EOF
```

---

## Editing the benchmark

**Models** — the `MODELS` list at the top of `harness/bench.py`: `(repo, port,
extra_server_args)`. Give each a distinct port.

**Tasks** — the `TASKS` list in `bench.py` (tsc repairs) or `bench_extended.py` (which
also supports `kind: "vitest"` and `kind: "noop"`, plus a `setup` hook that mutates the
fixture before the run — that is how T3 injects its bug).

Note the grader scores against a **baseline**: a task must clear its own target error
and introduce no new ones. Pre-existing errors in files a task doesn't touch are not
counted against the model. Getting this wrong understates results badly — my first
version required zero total errors and marked correct fixes as failures.

### Grading traps, learned the hard way

Three grader bugs in this project inflated scores. All were found by reading raw model
output; none were visible in the pass/fail column. If you extend the suite, check these:

1. **"Produced no edit" is not "correctly declined."** On a trap task these look
   identical. A model that exhausts its token budget mid-thought produced nothing, but
   it did not decide anything. Check `finish_reason == "length"` and score it separately.

2. **Strict anchor matching conflates two different failures.** One model dropped a line
   of source (a real content error); another emitted a semantically perfect edit indented
   4 spaces where the file used 2. Scoring both as "failed to reproduce the source" is
   wrong — every real applier normalises indentation. `bench_extended.py` matches
   leniently and records `anchor_match: "exact" | "lenient"` so you can still tell.

3. **Don't match a decline phrase anywhere in the output.** A model that quotes your
   instruction back ("reply with exactly: NO CHANGE NEEDED") while rambling will match.
   Require a completed response *and* the phrase near the end.

Superseded results are kept in `results/*-BADGRADER.json` rather than deleted.

**Token budget** — `MAX_TOKENS = 8000`. Thinking models need headroom to reason *and*
answer; too small a cap is consumed entirely by reasoning and returns empty content.

---

## Troubleshooting

### `SIGABRT` immediately, no weights loaded

```
[mpi] MPI found but it does not appear to be Open MPI. MLX requires Open MPI but this is MPICH
```

MLX resolved `libmpi.dylib` to Anaconda's MPICH. **`DYLD_LIBRARY_PATH` does not fix
this** and neither does reordering `PATH`. Use:

```bash
export MLX_MPI_LIBNAME=$(brew --prefix open-mpi)/lib/libmpi.dylib
```

`mlx_server.py` sets this automatically when Homebrew is present.

### `ValueError: Model type <x> not supported`

`mlx-lm` has no implementation for that architecture. Check before downloading with
`screen_config.py`. As of 0.31.3 there are 119 supported types; `cohere2_moe`, for
one, is not among them.

### Machine runs out of memory during a long prompt

Prefill, not KV cache. Lower `--prefill-step-size` (512 was both safer and faster
than the default 2048) and arm the watchdog.

### A model returns empty output

It is probably a thinking model putting everything in `reasoning_content` with no
`content` key. `mlx_server.chat()` returns both fields and `message_keys` so you can
confirm. Either raise `MAX_TOKENS` or disable thinking:

```bash
--chat-template-args '{"enable_thinking": false}'
```

### Results differ from mine

Expected across `mlx-lm` versions, quantizations, and model revisions. Within one
setup runs are deterministic — `mlx_lm.server` defaults to `--temp 0.0`, so identical
requests give byte-identical output. If your results move *within* one setup, check
whether your client is sending a temperature.
