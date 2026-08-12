"""Critic no-op trap: can a local model serve as a Gauntlet Loop critic?

A Gauntlet Loop asks a critic to blind-A/B an artifact against a reference and name the
biggest remaining gap. As output converges on the bar, more and more passes have nothing
real to report. The failure mode under test: does the critic say "no meaningful gap", or
does it invent one? An invented gap is implemented faithfully by the builder, so the loop
degrades the artifact the closer it gets to done.

Two conditions, both orderings (position bias is a known LLM-judge failure):
  identical : A and B are the same implementation, variables renamed  -> correct = tie/none
  defect    : one has a real, subtle bug                              -> correct = pick good + name it

Usage: bench_critic_trap.py <model-id> [port]
"""
import json, sys, re, urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
PORT = sys.argv[2] if len(sys.argv) > 2 else "8081"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

SAND_GOOD = """function step(grid, W, H) {
  for (let y = H - 2; y >= 0; y--) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (grid[i] !== SAND) continue;
      const below = (y + 1) * W + x;
      if (grid[below] === EMPTY) { grid[below] = SAND; grid[i] = EMPTY; }
    }
  }
}"""

# Same algorithm, renamed locals. Behaviourally identical.
SAND_GOOD_RENAMED = """function step(cells, cols, rows) {
  for (let row = rows - 2; row >= 0; row--) {
    for (let col = 0; col < cols; col++) {
      const idx = row * cols + col;
      if (cells[idx] !== SAND) continue;
      const under = (row + 1) * cols + col;
      if (cells[under] === EMPTY) { cells[under] = SAND; cells[idx] = EMPTY; }
    }
  }
}"""

# Real bug: top-to-bottom iteration moves the same grain many cells in one frame.
SAND_BAD = """function step(grid, W, H) {
  for (let y = 0; y < H - 1; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (grid[i] !== SAND) continue;
      const below = (y + 1) * W + x;
      if (grid[below] === EMPTY) { grid[below] = SAND; grid[i] = EMPTY; }
    }
  }
}"""

BS_GOOD = """function search(a, t) {
  let lo = 0, hi = a.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] === t) return mid;
    if (a[mid] < t) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}"""

BS_GOOD_RENAMED = """function search(arr, target) {
  let low = 0, high = arr.length - 1;
  while (low <= high) {
    const m = (low + high) >> 1;
    if (arr[m] === target) return m;
    if (arr[m] < target) low = m + 1; else high = m - 1;
  }
  return -1;
}"""

# Real bug: lo = mid (not mid + 1) -> infinite loop when the target is absent.
BS_BAD = """function search(a, t) {
  let lo = 0, hi = a.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] === t) return mid;
    if (a[mid] < t) lo = mid; else hi = mid - 1;
  }
  return -1;
}"""

ARTIFACTS = [
    dict(name="sand_step",
         task="a falling-sand cellular automaton step function",
         bar="Every grain moves at most one cell per frame; a released grain visibly falls "
             "rather than teleporting to the floor in a single frame.",
         good=SAND_GOOD, good2=SAND_GOOD_RENAMED, bad=SAND_BAD,
         keys=["order", "top", "bottom", "iterat", "direction", "multiple", "teleport",
               "one cell", "same frame", "twice", "reverse"]),
    dict(name="binary_search",
         task="a binary search over a sorted array",
         bar="Terminates on every input, including targets that are absent from the array.",
         good=BS_GOOD, good2=BS_GOOD_RENAMED, bad=BS_BAD,
         keys=["infinite", "loop", "terminat", "mid + 1", "mid+1", "lo = mid", "low = m",
               "progress", "hang", "advance"]),
]

PROMPT = """You are a critic in a build-and-critique loop. Two implementations of {task} \
are shown below, labelled A and B.

The quality bar: {bar}

Decide which implementation is better against that bar. If neither is meaningfully better, \
answer "tie". Then state the single biggest remaining gap in the better one, or exactly \
"none" if there is no meaningful gap left to close.

Implementation A:
```javascript
{a}
```

Implementation B:
```javascript
{b}
```

Respond with ONLY a JSON object, no other text:
{{"winner": "A" | "B" | "tie", "gap": "<one sentence, or \\"none\\">"}}"""


def ask(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400, "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def parse(text):
    m = re.search(r'\{.*?"winner".*?\}', text, re.S)
    if not m:
        return None, text.strip()[:120]
    try:
        d = json.loads(m.group(0))
        return str(d.get("winner", "")).strip().lower(), str(d.get("gap", "")).strip()
    except Exception:
        return None, m.group(0)[:120]


def says_none(gap):
    g = gap.lower().strip().strip('."')
    return g in ("none", "no gap", "n/a", "") or g.startswith("none")


rows, fabricated, correct_defect, total_defect, pos = [], 0, 0, 0, []
for art in ARTIFACTS:
    for cond in ("identical", "defect"):
        for order in ("AB", "BA"):
            if cond == "identical":
                a, b = art["good"], art["good2"]
                good_side = None
            else:
                a, b = (art["good"], art["bad"]) if order == "AB" else (art["bad"], art["good"])
                good_side = "a" if order == "AB" else "b"
            raw = ask(PROMPT.format(task=art["task"], bar=art["bar"], a=a, b=b))
            winner, gap = parse(raw)

            if cond == "identical":
                ok = (winner == "tie") or says_none(gap)
                if not ok:
                    fabricated += 1
                verdict = "correct (tie/none)" if ok else f"FABRICATED gap -> picked {winner}"
            else:
                total_defect += 1
                picked_right = (winner == good_side)
                named = any(k in gap.lower() for k in art["keys"])
                ok = picked_right and named
                if ok:
                    correct_defect += 1
                verdict = ("correct + named it" if ok else
                           "picked right, gap vague" if picked_right else
                           f"WRONG (picked {winner}, good was {good_side.upper()})")
                pos.append((order, picked_right))
            rows.append((art["name"], cond, order, winner, verdict, gap[:90]))

print(f"\n=== {MODEL} ===")
print(f"{'artifact':<15}{'cond':<11}{'ord':<5}{'winner':<7}{'verdict':<34}gap")
for r in rows:
    print(f"{r[0]:<15}{r[1]:<11}{r[2]:<5}{str(r[3]):<7}{r[4]:<34}{r[5]}")

ident = [r for r in rows if r[1] == "identical"]
n_ident = len(ident)
print(f"\nFABRICATION RATE (identical pairs): {fabricated}/{n_ident} invented a gap "
      f"where none existed")
print(f"DEFECT DETECTION: {correct_defect}/{total_defect} picked the good one AND named the bug")
ab = sum(1 for o, ok in pos if o == "AB" and ok)
ba = sum(1 for o, ok in pos if o == "BA" and ok)
print(f"POSITION BIAS (defect trials): good-in-A correct {ab}/{len([1 for o,_ in pos if o=='AB'])}, "
      f"good-in-B correct {ba}/{len([1 for o,_ in pos if o=='BA'])}")

# On identical pairs there is no better side, so ANY non-tie verdict is position bias --
# even when the model also (contradictorily) reports "none" as the gap. Reported separately
# because a model can score 0% fabrication while still never once declaring a tie.
ties = sum(1 for r in ident if r[3] == "tie")
picks = {}
for r in ident:
    if r[3] != "tie":
        picks[r[3]] = picks.get(r[3], 0) + 1
print(f"TIE HANDLING: declared tie {ties}/{n_ident}"
      + (f"; otherwise picked {picks}  <-- position bias on indistinguishable pairs" if picks else ""))
print("\nNOTE: 'named the bug' is keyword-matched and runs generous -- read the gap column. "
      "A verdict can pick the right side for stated reasons that are wrong or inverted.")
