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
