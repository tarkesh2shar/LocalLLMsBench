# LinkedIn drafts

Three options. All link to https://github.com/tarkesh2shar/LocalLLMsBench

---

## Option A — the hallucination hook (recommended, ~230 words)

I benchmarked 9 local coding models on a 48 GB Mac. The rankings were the least
interesting thing I found.

One task kept defeating them. A TypeScript error: `'error' is declared but its value
is never read`, at line 149. A one-line fix.

8 of 9 configurations failed it. So I asked one to think out loud:

"The error refers to line 149, which is `const error = localStorage.getItem('error')`"

That line doesn't exist. Anywhere. The entire file was in its context window, and it
invented the line — then reasoned flawlessly from a fabricated premise.

The one model that got it right did something different:

"line 149,16. Let's count lines? But easier: In handleCitySelect, we have catch..."

It refused to resolve the line number and found the code semantically instead.

That's the whole difference. Not parameters, not benchmarks — one model doesn't trust
line numbers and the others do.

Two follow-ups:

→ I added chain-of-thought prompting to fix it. The model got WORSE: 2/4 → 0/4. CoT
launders a hallucination into a confident, well-argued wrong answer.

→ I tried higher precision. 6-bit produced byte-identical output to 5-bit. Not a
quantization problem.

If you're routing work to a local model: don't send "error at line N". Send the
function name and the actual snippet.

Harness, fixture and raw data are public — 2 tasks, one repo, so treat the numbers as
a starting point, not gospel.

---

## Option B — the gotchas hook (~200 words)

Three things silently broke my local LLM benchmark before model quality mattered at
all. None are documented well anywhere I could find.

1. MLX aborts if it finds the wrong MPI.

`mlx_lm.server` dlopen()s libmpi.dylib. If that resolves to Anaconda's MPICH instead
of Open MPI, it SIGABRTs before loading a single weight. DYLD_LIBRARY_PATH does NOT
fix it — I verified the variable reaches the process. The knob that works is
MLX_MPI_LIBNAME.

2. Prefill is the memory event, not the KV cache.

A 74K-token prompt drove a 48 GB machine to within 4 GB of full — twice — while
resident memory sat flat at 18.6 GiB. It's prefill activation buffers. Dropping
--prefill-step-size from 2048 to 512 made it both safer AND faster: peak available
memory went 6→14.7 GiB, throughput 410→750 tok/s.

3. "sliding_window: 128" doesn't mean short memory.

Common advice says a small sliding window disqualifies a model for long context.
gpt-oss-20b has a 128-token window and recalled a token planted 20,000 tokens back —
because 12 of its 24 layers are full attention. Screen on layer_types, not
sliding_window. The naive check would reject gpt-oss, Gemma and Mistral.

Full write-up and harness:

---

## Option C — short and punchy (~110 words)

Benchmarked 9 local coding models on a 48 GB Mac. Three findings that surprised me:

→ Models hallucinate what's at "line 149" — with the file in their context. The one
model that solved my hardest task did so by REFUSING to count lines and locating the
code by function name instead.

→ Chain-of-thought made the coder model worse. 2/4 → 0/4. It turns a fabricated
premise into a confident, well-argued wrong answer.

→ The #1 open-source model on SWE-bench came last. SWE-bench measures multi-turn work
with test feedback. That's a different skill from one-shot bounded edits.

Harness, fixture, and raw results are public. 2 tasks on 1 repo — a starting point,
not a verdict.

---

## Notes on posting

- LinkedIn shows ~3 lines before "see more". All three drafts front-load the hook.
- LinkedIn suppresses reach on posts with outbound links. Consider putting the repo
  link in the first comment and saying "link in comments".
- No hashtag soup. If you want any: #LocalLLM #AppleSilicon #MLX
- Option A is the strongest — a concrete surprise with a mechanism and a takeaway.
  Option B will land better with an infra/systems audience.
- Keep the "2 tasks, one repo" caveat in whichever you use. It costs one line and
  removes the obvious line of attack.
