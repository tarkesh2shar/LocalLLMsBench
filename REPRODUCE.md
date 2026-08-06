# Reproducing the benchmark

Everything here was run on macOS (Apple Silicon) with `mlx-lm` 0.31.3.

## 1. Prerequisites

```bash
brew install open-mpi llama.cpp
uv tool install mlx-lm
```

`llama.cpp` is only needed for round 4 (models with no MLX build, e.g. Bonsai).

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

### Round 3: speculative decoding (graded, not just timed)

```bash
python3 harness/bench_specdec.py
```

Runs the five tasks three ways — no drafter, drafter with 3 draft tokens, drafter with
5 — plus a "count from 1 to 20" canary that makes token corruption obvious.

Deliberately graded rather than timed, because corrupted output fails `tsc`/`vitest`.
That measures whether speculative decoding is *usable*, not merely faster.

**Check the pairing before you run it.** A drafter must satisfy three conditions, and
Qwen3.6-27B fails all three:

| Requirement | Check |
|---|---|
| drafter architecture implemented by mlx-lm | `screen_config.py` on the **drafter**, not just the target |
| target's KV cache is trimmable | hybrid-attention models use `ArraysCache` — rejected |
| matching vocab | `vocab_size` must be identical, not merely "same family" |

```bash
python3 -c "
import json,glob
for m in ['Qwen3-Coder-30B-A3B-Instruct-5bit','Qwen3-0.6B-4bit']:
    f=glob.glob(f'{__import__(\"os\").path.expanduser(\"~\")}/.cache/huggingface/hub/models--mlx-community--{m}/snapshots/*/config.json')[0]
    c=json.load(open(f)); t=c.get('text_config',c)
    print(m, c['model_type'], t.get('vocab_size'))"
```

### Round 4: llama.cpp — isolating runtime from quantization

```bash
brew install llama.cpp
python3 harness/bench_llamacpp.py qwen3.6    # control
python3 harness/bench_llamacpp.py bonsai     # 1-bit
```

`llama_server.py` starts `llama-server` and reuses the same OpenAI client, so the tasks
and grading are unchanged across runtimes.

Run the **control first**. Comparing an MLX 4-bit model against a llama.cpp 1-bit model
changes two variables at once; without the same-quantization control you cannot tell
compression from runtime. Measured here: 14.8 vs 15.3 tok/s, 3/5 both — runtime neutral.

One invocation per model keeps a single model resident. Results **merge** into
`results-llamacpp.json` rather than overwriting (an earlier version overwrote, and
destroyed the control arm's data).

### Downloading GGUFs

`huggingface_hub` stalled at 68 KB/s here while `curl` sustained 12 MB/s to the same
URL — a 175× difference. If HF downloads crawl:

```bash
curl -L -C - -o model.gguf "https://huggingface.co/<repo>/resolve/main/<file>.gguf"
```

Repos often hold every quantization (80 GB for all of `Ternary-Bonsai-27B-gguf`), so
fetch **single files**, not snapshots.

**Verify after download.** A resumed transfer can produce a full-size but corrupt file:

```bash
head -c 4 model.gguf   # must be "GGUF"
llama-cli -m model.gguf -p hi -n 1 --no-warmup
```

A `tensor '<name>' has offset X, expected Y` error on a *freshly downloaded,
size-verified* file is not corruption — it means the quant type's block layout differs
from what your llama.cpp build expects. `Ternary-Bonsai-27B-Q2_0.gguf` fails this way
on upstream llama.cpp despite `Q2_0` appearing in `llama-quantize --help`. A quant name
being listed does not mean an arbitrary file claiming that name will load.

### Round 5: structured handoff (multi-turn agent loop)

```bash
python3 harness/e4_handoff.py
```

The only multi-turn experiment here. Builds a repo copy with 3 failing tests across 2
files and runs two arms — a vague "another agent worked on this, continue" against an
explicit brief naming the failing tests and the source files — measuring context at first
edit, turns, and whether the tests end up green.

**The model never gets a shell.** It has a fixed verb set (`LIST` / `READ` / `TEST` /
`EDIT` / `DONE`) confined to the repo copy, with path-escape blocked and test files
refused. Malformed commands return an error rather than crashing the loop.

Two harness details that matter if you extend it:

- **Parse every verb's argument from the first line only.** `cmd[4:]` swallows the whole
  multi-line command into the filename — that produced a `File name too long` crash and,
  before that, silent `LIST` failures.
- **Use the non-streaming endpoint.** `mlx_lm.server` does not return `prompt_tokens` in
  SSE chunks, and context-per-turn is the entire measurement — streaming reports 0.

Results in `results/results-e4.json`, including the full per-turn command trace.

### Round 6: diff vs whole-file across turns

```bash
python3 harness/e5_editmode.py
```

Same task and brief as round 5's winning arm; the only difference is the edit verb the
model is given — `EDIT` with a SEARCH/REPLACE block, or `WRITE` with the complete file.

`WRITE` records how much the file shrank and warns above 30%, because a truncated
whole-file reply *replaces* the file — the failure anchor edits structurally cannot have.

Results in `results/results-e5.json`.

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
