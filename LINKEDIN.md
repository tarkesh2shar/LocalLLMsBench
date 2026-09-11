# LinkedIn drafts

LinkedIn does not render Markdown. No `**bold**`, no `#` headings, no `-` bullets that
turn into lists. What works: short lines, blank lines between them, and `→` or `•` as
manual bullets.

Everything below is copy-paste ready. Do not add formatting.

Repo: https://github.com/tarkesh2shar/LocalLLMsBench

---

## OPTION A — "I published the wrong recommendation" (recommended)

~1,900 characters. Lead is the strongest thing we have: a public correction.

```text
I benchmarked 9 local coding models on a 48GB Mac, published a recommendation, then
added 3 more tasks.

The ranking inverted.

Round 1 was two real TypeScript build errors. Qwen3.6-35B swept it 4/4. Qwen3-Coder-30B
scored 2/4. I wrote that Qwen3-Coder should be retired as a candidate.

Then I added three task kinds:

→ Fix a runtime bug from a failing test
→ Implement a function against a spec test
→ A clean file plus an error message I made up

Qwen3-Coder scored 5/5 on the new tasks.
Qwen3.6-35B dropped to 3/5.

Combined, they tie at 7/9 — but Qwen3-Coder used 5.4x fewer tokens and 5.3x less time.

The model I told people to retire is the one I would actually deploy.

Two tasks were not enough. My confident recommendation was wrong, and it took 40 more
runs to find out.

The third task turned out to be the interesting one. Give a model a clean file and an
error that does not exist, and only 2 of 8 handled it. The rest either burned their
entire token budget searching (one spent 485 seconds per attempt), or invented a fix
and edited a file that had nothing wrong with it.

One model, caught mid-reasoning, was numbering the lines of the file one by one trying
to reach line 42. It ran out of budget before it got there.

If you are routing work to a local model: a stale or mistaken error message does not
produce a fast "not found". It produces a multi-minute hang, or a confident edit to
working code.

I also found three bugs in my own grader, each of which inflated scores. All three were
found by reading raw model output. None were visible in the pass/fail column.

Harness, fixture, raw results and the superseded numbers are all public. Five tasks on
one repo — still a starting point, not a verdict.

https://github.com/tarkesh2shar/LocalLLMsBench
```

---

## OPTION B — the infrastructure gotchas

~1,500 characters. Better for a systems/infra audience. These findings were the hardest
to find documented anywhere.

```text
Three things silently broke my local LLM benchmark before model quality mattered at all.

1. MLX aborts if it finds the wrong MPI

mlx_lm.server dlopen()s libmpi.dylib. If that resolves to Anaconda's MPICH instead of
Open MPI, it SIGABRTs before loading a single weight.

DYLD_LIBRARY_PATH does not fix this. I verified the variable reaches the process.
Reordering PATH does not fix it either.

The knob that works is MLX_MPI_LIBNAME.

2. Prefill is the memory event, not the KV cache

A 74K-token prompt drove a 48GB machine to within 4GB of full, twice, while the server's
resident memory sat flat at 18.6 GiB. It is prefill activation buffers, not accumulated
cache.

Dropping --prefill-step-size from 2048 to 512 made it both safer AND faster:
→ available memory during prefill: 6 GiB to 14.7 GiB
→ throughput: 410 to 750 tok/s

If you model memory as weights + kv_per_token x context, you are missing the term that
actually binds.

3. "sliding_window: 128" does not mean short memory

Common advice says a small sliding window disqualifies a model for long context.

gpt-oss-20b has a 128-token window and recalled a token planted 20,000 tokens back,
because 12 of its 24 layers are full attention.

Screen on layer_types, not sliding_window. The naive check rejects gpt-oss, Gemma and
Mistral.

Full write-up, harness and raw data:
https://github.com/tarkesh2shar/LocalLLMsBench
```

---

## OPTION C — short

~700 characters. For low effort or as a follow-up post.

```text
Benchmarked 9 local coding models on a 48GB Mac. Three things I did not expect:

→ I published a recommendation after 2 tasks. Added 3 more and the ranking inverted.
The model I said to retire scored 5/5 and now uses 5x fewer tokens than the one I
recommended.

→ Give a model a clean file and a fabricated error, and 6 of 8 either hang for minutes
or invent a fix and edit working code. One was numbering lines one by one trying to
reach line 42.

→ One model answered a "return the whole file" request with 57 tokens. Applied, it
wiped 100 lines. The same model fixed the same bug via search/replace in 41 tokens.

Harness and raw data:
https://github.com/tarkesh2shar/LocalLLMsBench
```

---

## ROUND 8 — Muse Glimmer 30B (2026-08-12)

### OPTION A — "best score, still not deploying it" (recommended)

~2,100 characters. The tension is real and it is the honest read.

```text
A 30B model running in 12GB of RAM just posted the best score my local benchmark has
ever recorded.

I am still not making it my default.

Meta shipped Muse Glimmer 30B open on August 10. Unsloth's 2-bit build is 12.4GB on
disk. I ran it on an M5 Pro with 48GB unified memory, through llama.cpp on Metal.

12.3 GiB resident. 20 tokens/sec. Nine graded tasks: fix a type error, fix a runtime
bug from a failing test, implement against a spec, and a trap where the file is clean
and the error is one I made up.

9/9 sampled. 8/9 greedy.

Previous best was Seed-OSS-36B at 8/9. My current daily driver, Qwen3-Coder-30B, scores
7/9.

It also solved the one task that had beaten everything.

A file where "error" is both a useState variable and a catch parameter. The compiler
points at line 149. Every model that failed it fabricated the contents of that line and
then reasoned correctly from an invented premise. Qwen3-Coder failed it in every
configuration I tried — plain, chain-of-thought, 5-bit, 6-bit.

Glimmer got it. Both edit strategies, sampled and greedy.

So why am I not switching?

→ 872 seconds vs 99 for the same five tasks. Roughly 9x.
→ On a multi-turn repair loop: 333 seconds vs 11.5. Roughly 29x.
→ Hand it failing tests without file names and it fixed one bug, then re-ran the test
suite 24 times in a row until it hit the turn cap. Qwen3-Coder solves that in 9 turns.

Cost is zero on local hardware. Wall clock is the entire budget. A 9x tax buys two
graded points.

Where it does win outright: 13 KiB per token of KV cache against Qwen3-Coder's 96. That
is 7x more context headroom on the same machine, and it changes how many agents you can
run at once.

And the tool calling is genuinely clean — 52 of 52 turns emitted well-formed structured
calls, zero malformed arguments. It ran the test suite to confirm the bug was real
before touching a file. None of the 8 models I tested previously did that.

Then I gave it a bug that does not exist.

It read the code, ran the suite, saw 26 passing tests, changed nothing — and reported
the job complete.

No damage. No invented fix. Just a worker that says done and hands back an empty diff.
Qwen3-Coder reaches the same place by a worse route: it emits a search/replace where
both halves are byte-identical, then declares success.

Different models. Different failure. Same ledger entry.

If you are building on local agents: do not trust "done". Diff the result.

One repo, one run per arm, 2-bit only. Vision and prompt injection untested.

https://github.com/tarkesh2shar/LocalLLMsBench
```

---

### OPTION B — the reproducibility correction

~1,400 characters. Systems audience. This one stings and is the most useful.

```text
I found out half my benchmark was never reproducible, 40 runs after I published it.

I have been comparing local models on a 48GB Mac across two servers: mlx_lm.server for
models with MLX builds, llama-server for the ones without.

mlx_lm.server defaults to temperature 0.0. Greedy. Identical prompt, identical output.

llama-server defaults to temperature 0.8, top_k 40, top_p 0.95, random seed.

Neither of my wrappers sends a sampling parameter. So every MLX run was deterministic
and every llama.cpp run was sampled, and I had labelled all of them reproducible.

I caught it because the same prompt replayed at 3,658 then 4,313 completion tokens.

Check GET /props. Do not assume the default is greedy.

Two more from the same run:

1. Budget agent turns generously for a reasoning model

At 1,200 tokens per turn my tool-calling loop looked broken — turns arriving with no
tool call at all.

It was not a tool-calling failure. The turn was truncating, and after a truncated turn
the model started writing a sentence before its call. llama.cpp parses the call when it
is the whole message, not when prose precedes it. The call arrived structurally perfect
and sitting in the wrong field.

At 3,000 tokens: 52 of 52 parsed.

Treat finish_reason: length in an agent loop as a hard error. A truncated turn corrupts
the shape of the turns after it.

2. My own grader scored a correct fix as 0/4

The runner never populated its baseline error set, so a pre-existing error in a file the
model never opened counted as damage. The textbook correct fix graded as a failure.

That is the fourth grader bug in this project. Every one was found by reading raw model
output. None were visible in the pass/fail column.

https://github.com/tarkesh2shar/LocalLLMsBench
```

---

### OPTION C — short

~800 characters.

```text
Meta's Muse Glimmer 30B, 2-bit, running in 12GB on an M5 Pro with 48GB unified memory.

Best score my local benchmark has recorded: 9/9 sampled, 8/9 greedy. Previous best 8/9.
Qwen3-Coder-30B, my daily driver, is 7/9.

It solved the task nothing else could — a shadowed identifier where every other model
hallucinated the contents of the line the compiler pointed at.

I am still not switching.

→ 9x slower single-shot, 29x on a multi-turn repair loop
→ 20 tok/s against Qwen3-Coder's 65
→ Give it failing tests without file names and it loops the test suite 24 times

What it wins: 13 KiB/token of KV cache against 96. Seven times the context headroom.

And handed a bug that does not exist, it changed nothing and reported the job complete.
The better model still hands you an empty diff and calls it done.

https://github.com/tarkesh2shar/LocalLLMsBench
```

---

## OPTION D — "A video claimed this model is 2.7x faster. I tested it."

~1,850 characters. Highly engaging hook dissecting the "2.7x" benchmark claim with empirical data.

```text
A video claimed a new fine-tuned Qwen 3.8-27B model ("Dirk") is "2.7x faster" for local coding.

I downloaded the 16 GB weights to my Mac to see if the claim is real.

The short answer: yes, but not the way you think.

If you measure raw token generation on Apple Silicon:
• Stock Qwen 3.8: 22.8 tok/s
• Dirk-Qwen: 23.1 tok/s

The model does not magically double your memory bandwidth. Hardware decode speed is identical.

Yet when I benchmarked them on identical TypeScript compiler repair tasks:
• Stock Qwen 3.8 took 543.5 seconds (~9 minutes) and burned 13,274 tokens.
• Dirk finished in 156.8 seconds (~2.5 minutes) and used only 3,482 tokens.

Both solved 4 out of 4 tasks with clean compiler passes.

Dirk was 3.5x faster in real wall-clock time, using 74% fewer tokens.

Why?

Stock Qwen 3.8 defaults its reasoning effort to "xhigh". When given a 2-line bug fix, it spends 3,000 to 5,000 internal tokens rambling in <think> tags before writing code.

Dirk ships with a custom "Sharp" chat template. It forces terseness into the system prompt and defaults thinking to "medium". It doesn't generate tokens faster — it stops generating tokens you never asked for.

Two caveats I found while stress-testing it:

1. Turn thinking completely off, and it runs in 124 seconds (4.4x faster), but it fails subtle trap tasks — it tries to edit clean code when given a fake bug.

2. The sweet spot was setting reasoning effort to "low": 100% 5/5 pass rate across real bugs, spec implementations, and no-op traps, in less than half the time of the base model.

Also confirmed: native MTP speculative decoding gives a +65% pure decode speedup (14 tok/s → 23 tok/s) on Apple Silicon Metal with zero degradation.

Raw traces, prompts, and reproduce scripts are open:

https://github.com/tarkesh2shar/LocalLLMsBench
```

---

## OPTION E — "65 tokens/sec on Mac, but speculative decoding slows it down"

~1,850 characters. Contrasting Dense vs MoE local coding models and the counterintuitive MTP inversion.

```text
I just benchmarked the newest local MoE model ("Nail", based on Qwen 35B-A3B) on Apple Silicon.

It revealed two things I did not expect:

1. The raw decode speed is 65 tokens per second.
2. Speculative decoding actually made it slower.

Here is the comparison against the dense 27B model (Dirk) running on the same Mac:

• Dirk-27B (Dense): 14 tok/s baseline → 23.1 tok/s with MTP (+65% speedup)
• Nail-35B (MoE): 65.1 tok/s baseline → 55.1 tok/s with MTP (15% SLOWER)

Why does speculative decoding invert on MoE?

On a dense 27B model, every forward pass reads 15 GB of weights. It is heavy, so generating draft tokens and batch-verifying them saves massive memory bandwidth.

On a 35B-A3B MoE, only 3.39B parameters are active per token. Generating a token is so cheap and fast (65 tok/s) that the overhead of draft verification and context management actually slows it down.

If you are running lightweight MoEs locally: do not enable MTP speculative decoding. Plain baseline is fastest.

Then came the task benchmarks.

On real TypeScript build errors, Nail swept 4/4 in 175 seconds. It is blazing fast and accurate.

Then I handed it the no-op trap: a clean, working file, with an error message I completely fabricated.

The dense model (Dirk) saw right through it: declined the edit and passed.

The MoE model fell into an architectural rabbit hole: it literally spent 15,499 tokens counting lines one by one trying to find line 42 in the code:
"1: // import... 2: (empty)... 3: export const... 12: Drizzle..."

It burned its entire context window and crashed on the token cap.

MoEs give you server-grade 65 tok/s speed on a 16GB–48GB Mac. But dense models still win hands-down when dealing with false premises and hallucinations.

Full benchmarks, logs, and reproduction scripts:

https://github.com/tarkesh2shar/LocalLLMsBench
```

---

## Posting notes

**Formatting**
- LinkedIn renders none of Markdown. Paste as plain text.
- Blank line between every 1-2 lines. Dense paragraphs get skipped.
- `→` and `•` survive paste. Asterisk bullets do not become lists.
- First ~3 lines show before "see more" — all three drafts front-load the hook.

**The link**
- LinkedIn suppresses reach on posts with outbound links. Two options:
  1. Post as-is and accept the hit (simplest, link is visible)
  2. Remove the URL, end with "Link in comments", then immediately comment it
- Option 2 reliably reaches further. Comment within the first minute.

**Hashtags** — optional and low value. If any: #LocalLLM #MLX #AppleSilicon

**Which to post**
- A is strongest. A public "I was wrong" with numbers is rare and it is the real story.
- B if your audience is infra/systems — those three findings are the most novel.
- C as a follow-up a few days later, linking back.

**Keep the caveat.** The "five tasks on one repo" line costs you nothing and removes the
one fair criticism. Losing it is how a post gets picked apart in the comments.
