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

### Round 7: a false objective in a multi-turn loop

```bash
python3 harness/e5e_trap_multiturn.py --setup      # build the two fixtures
python3 harness/e5e_trap_multiturn.py --suite=all
```

Five of the eight arms run against a repo where **all 26 tests already pass** and the
brief is wrong — either a bug that has never existed, or round 5's brief verbatim against
a repo where every bug it names is already fixed. Plus a positive control and an
isolation of *which block* of the brief carries the effect.

`--setup` derives `e5e_workspace/{buggy,healthy}` from round 5's pristine fixture and
refuses to continue unless they report 3 and 0 failures respectively.

Three things this harness records that rounds 5 and 6 did not, each because of a real bug:

- **`finish_reason` on every completion.** On a trap task "produced no edit" and
  "correctly declined" are indistinguishable without it.
- **The full raw reply per turn**, not `reply.split("\n")[0][:80]`. Rounds 5 and 6 were
  truncating away a stray `>>>>>>> REPLACE` the model appended to nearly every command;
  a `TEST` costing 5 tokens instead of 2 was the only trace of it.
- **No-op edits as a distinct kind.** Every trap arm emitted an `EDIT` whose SEARCH and
  REPLACE bodies were byte-identical. A permissive applier answers `Edited <file>.` and
  the model builds on that false success. `_noopdet` arms answer honestly instead — it
  did not change the outcome, which is itself the result.

Two traps specific to writing this kind of arm:

1. **Do not give an escape-hatch verb an example.** The first version documented
   `REPORT` with `Example: REPORT blocked, the build tooling is not installed`. The model
   emitted that sentence verbatim to abandon a real, half-finished repair. Remove the
   example and the same arm solves the task.
2. **Keep the system prompt's tail byte-identical to round 5's.** Inserting the new verb
   *between* the EDIT block and `DONE` made the model append 1,200 tokens of repeated
   `>>>>>>> REPLACE` after every command. Inserting it *above* EDIT — same text, same
   verbs — dropped that to 7 tokens with an identical decision trace.

Results in `results/results-e5e.json`; the superseded first round is kept in
`results/results-e5e-round1.json`.

### Round 8: native tool calling, and a model with no MLX build

Muse-Glimmer-30B (Meta, 2026-08-10) is the first candidate whose claimed strength is
tool calling rather than coding, and `mlx-lm` 0.31.3 cannot load it (`model_type:
muse_glimmer`), so it runs on llama.cpp only. Two harnesses exist because of that.

```bash
# 1. screen the config before downloading 12 GB
python3 harness/screen_config.py meta-models/Muse-Glimmer-30B

# 2. the five vitest/tsc-graded tasks (starts and stops its own llama-server)
python3 harness/bench_llamacpp.py glimmer

# 3. T1/T2 against a server you started yourself
llama-server -m ~/models/gguf/Muse-Glimmer-30B-UD-Q2_K_XL.gguf \
  --host 127.0.0.1 --port 8095 -c 32768 -ngl 999 --no-webui --jinja \
  --temp 0 --top-k 1 --parallel 1
python3 harness/bench_t12_external.py --port 8095 --label glimmer

# 4. the same fixtures and briefs as round 7, but with native OpenAI `tools`
python3 harness/bench_toolcall.py --port 8095 --label glimmer --max-tokens 3000
```

`bench_t12_external.py` exists because `bench.py` owns its own MLX server, so a model
with no MLX build could not be scored on T1/T2 at all. **It must populate
`bench.BASELINE` itself** — that is filled inside `bench.main()`, and leaving it empty
makes the *other* pre-existing error, in a file the model never touched, count as
damage. That scored a canonical correct fix as 0/4 on the first run.

`bench_toolcall.py` changes exactly one variable against round 7: the model is handed
OpenAI-format `tools` and must emit structured `tool_calls` instead of text commands.
It adds a `malformed_calls` metric a regex-parsed protocol cannot express.

Two things will bite you:

1. **Budget agent turns at 3,000+ tokens for a reasoning model.** At 1,200 a turn
   truncates (`finish_reason: length`), and after a truncated turn the model starts
   emitting a commentary preamble, `<|eom|>`, then a second message carrying the call.
   llama.cpp parses that header when it is the whole message but not when a preamble
   precedes it, so the call arrives structurally perfect and sitting in `content` with
   `tool_calls: []`. At 3,000 tokens this vanishes entirely: 52/52 turns parsed. Treat
   `finish_reason: length` in an agent loop as a hard error — a truncated turn corrupts
   the shape of the turns after it. `--recover` enables a fallback extractor for the
   truncated case; it is off by default because it measures around a real problem.
2. **Raise `-c`.** A trap arm's re-verification loop ran itself out of a 16K context
   (`request (16458 tokens) exceeds the available context size`). 32K is enough.

Results in `results/results-toolcall-*.json` (full per-turn output and tool calls
retained), `results/results-t12-muse-glimmer-*.json`, and the `muse-glimmer-*` arms of
`results/results-llamacpp.json`.

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

Expected across `mlx-lm` versions, quantizations, and model revisions.

**Determinism depends on which server you are running, and the two disagree.**
`mlx_lm.server` defaults to `--temp 0.0`, so MLX runs are greedy and identical requests
give byte-identical output. **`llama-server` does not**: its defaults are temp 0.8 /
top_k 40 / top_p 0.95 / min_p 0.05 with a random seed — check with `GET /props`. Neither
`llama_server.start_server` nor `mlx_server.chat` sends a sampling parameter, so every
llama.cpp arm in this repo ran sampled unless its label says `-greedy`. Pass
`--temp 0 --top-k 1` to reproduce those.

This was caught late, by an identical prompt replaying at 3,658 then 4,313 completion
tokens. If your results move *within* one setup, check the server's defaults before
your client's.

## Reasoning budget (Qwen3.8 and any model with `reasoning_effort`)

Qwen3.8's chat template defaults `reasoning_effort` to **`xhigh`**. That default is
not neutral and it is not visible in the output, because `bench.py` strips `<think>`
blocks after generation — the tokens are still paid for. Every §16 number was
measured this way.

Override it per request, not per server:

```bash
python3 harness/bench_effort.py --port 8095 --suite t12      --effort off
python3 harness/bench_effort.py --port 8095 --suite extended --effort off
```

`--effort off` sends `{"enable_thinking": false}`; `low` / `medium` / `xhigh` send
`{"reasoning_effort": ...}`. On the five-task suite this is 5/5 in 96.7s against 229s
at the default, with the traps intact (§17).

**If you are benchmarking any thinking model, record the budget you ran at.** A score
compared across two different budgets is not a comparison.

## Decode rate on its own

```bash
python3 harness/bench_decode_rate.py --port 8095 --label "n-max=3 (default)"
```

Fixed prompt, fixed cap, greedy — so the `sha` in the output is the check that two
arms really are producing the same tokens at different speeds. If two arms disagree
on `sha`, you are comparing outputs, not speeds, and the tok/s numbers are not
comparable.

Caveat: `mlx_lm.server` returns neither `content` nor `reasoning_content` when the
reasoning block is still open at the token cap, so MLX arms hash the empty string.
`completion_tokens` is still authoritative there.

## `mlx_lm.server` exits immediately

If anaconda is on `PATH` before Homebrew, MLX finds anaconda's MPICH and refuses to
start, logging MPICH build details and nothing else — no traceback. `mlx_server.py`
already handles it; outside this harness set it yourself:

```bash
export MLX_MPI_LIBNAME=$(brew --prefix open-mpi)/lib/libmpi.dylib
```

## DFlash 2 drafter status and reproduction

DFlash 2 drafters (`z-lab/Qwen3.8-27B-DFlash2-GGUF`) can be downloaded via:

```bash
python3 download_dflash2.py
```

Currently, `llama.cpp` (Metal) expects 81 tensors (`draft-dflash` v1) whereas DFlash 2 provides 58 tensors, causing an init failure (`wrong number of tensors; expected 81, got 58`). For Apple Silicon, use native MTP (`--spec-type draft-mtp`) on `Qwen3.8-27B-Q4_0.gguf` until DFlash 2 tensor schema support is merged upstream.

