# Retrieval improvement experiment

Investigates the dominant gap found by the grader module: `recall_lenient`
(95.7%) vs. `recall_strict` (69.6%) — the AI usually states the current fact
correctly but frequently can't state the prior-period comparison that makes
it "net new." Confirmed directly for the CRWD Nov 26 case via the AI's own
`limitations` field ("financial figures from Q2 FY2025... are not present in
the retrieved prior excerpts"). This investigation asks: is that a retrieval
problem, and if so, how much of it is fixable?

All work here is **code-only, zero model calls, $0 cost** — every step
measures itself against ground truth pulled directly from
`references/development.jsonl`'s own `prior_evidence` citations, before
spending anything on a real replay + judge cycle. Nothing in this
investigation has touched `net_new/` — every script lives in `evaluation/`
and calls the real `net_new` functions (`retrieve()`, `build_input()`'s
selection logic, etc.) without modifying them, so these are all still
candidate ideas, not yet promoted into the pipeline.

## Method used throughout

For each of the **21 `prior_evidence` citations** across the 9 development
cases (23 reference findings; not all have prior evidence):

1. Locate which real history chunk actually contains that exact quote
   (ground truth — independent of any retrieval algorithm).
2. Run the retrieval logic being tested for that filing.
3. Check whether the ground-truth chunk made the result.

`ground truth located: 21/21` in every run below — every citation is a real,
findable fact somewhere in the un-truncated history. The only question is
whether the retrieval step actually surfaces it.

## Results

| Step | Method | Retrieved | Rate |
|---|---|---|---|
| **Step 0** | Baseline — real `retrieve()`, one merged query per filing, raw word-overlap count | 4/21 | 19.0% |
| **Step 1** | Query `retrieve()` independently per current-chunk (same scoring), merge via Reciprocal Rank Fusion | 9/21 | 42.9% |
| **Step 2** | Original merged query, IDF-weighted overlap scoring instead of raw count | 3/21 | **14.3% (worse than baseline)** |
| **Step 3** | Step 1 + recency-weighted scoring (exponential decay, half-life 180 days) | 11/21 | 52.4% |
| **Step 3 + K=24** | Step 3 with `history_chunks` doubled from 12 to 24 | 12/21 | 57.1% |

Files: [evaluation/retrieval_check.py](../evaluation/retrieval_check.py),
[evaluation/retrieval_check_step1_perchunk.py](../evaluation/retrieval_check_step1_perchunk.py),
[evaluation/retrieval_check_step2_idf.py](../evaluation/retrieval_check_step2_idf.py),
[evaluation/retrieval_check_step3_recency.py](../evaluation/retrieval_check_step3_recency.py),
[evaluation/retrieval_check_step3_k24.py](../evaluation/retrieval_check_step3_k24.py),
each with a matching JSON result under `evaluation/results/`.

### Step 0 — Baseline diagnostic

Ran the actual pipeline logic (`build_input()`'s query construction +
`net_new.parsing.retrieve()`), unmodified, against all 21 citations. **7 of 9
cases got zero correct retrievals.** This turned "the model said it couldn't
find it" into a direct, quantified fact instead of an inference from 3 cases'
worth of self-reported limitations.

### Step 1 — Per-chunk queries

**Hypothesis:** one merged query per filing dilutes a specific fact's search
signal with unrelated text elsewhere in the same document.

**Change:** query `retrieve()` once per included current-chunk instead of
once per filing; merge the results.

**A real bug found along the way:** the first merge implementation
(round-robin — best-of-chunk-0, then best-of-chunk-1, ...) filled the 12-slot
budget by *iteration order*, not match strength. Traced one case directly:
a ground-truth chunk was the **#1-ranked match** for one specific
current-chunk's query, but that chunk happened to be 12th in iteration order,
so the budget was already full by its turn. First-pass result under this
buggy merge was 7/21 (33.3%) — understated. Fixed with **Reciprocal Rank
Fusion** (score each candidate by summing `1/(rank+1)` across every
per-chunk list it appears in, then take the true global top-12). Corrected
result: **9/21 (42.9%)**.

### Step 2 — IDF-weighted scoring (negative result)

**Hypothesis:** down-weight boilerplate words that appear in nearly every
history chunk, up-weight rare/distinctive ones.

**Change:** kept Step 0's original single merged query; replaced raw
word-overlap count with smoothed IDF-weighted overlap
(`log((N+1)/(df+1)) + 1` per shared word, summed).

**Result: 3/21 (14.3%) — worse than doing nothing.** A genuine negative
result, not a bug (verified the IDF math is a correct, standard formula).
**Why it backfired:** IDF also up-weights *rare-word coincidences* — a
proper noun or odd phrase shared by chance with an unrelated boilerplate
chunk — while suppressing common-but-genuinely-relevant vocabulary
("revenue," "ARR," "guidance") that weakly but correctly connects two
same-topic disclosures. Dropped; not combined with anything else.

### Step 3 — Per-chunk queries + recency weighting

**Diagnosis that motivated this:** looked at what was *still* missed across
Steps 0-2 and found a pattern — many remaining citations point at content
that repeats near-identically across many history chunks (quarterly
comparison tables with the same headers every quarter; standard legal
disclaimers). The problem there isn't vocabulary, it's picking the
*temporally right* occurrence among several similar ones — a signal
(`available_at`, already on every document) nothing so far had used.

**Change:** Step 1's per-chunk querying, unchanged; multiply each chunk's
raw overlap score by `(1 + freshness_bonus)`, where `freshness_bonus =
exp(-days_ago / 180)` — full boost same-day, decaying toward zero over
roughly a year. A chunk with zero word overlap stays at zero regardless of
recency; recency only breaks ties among chunks that already share real
vocabulary.

**Verification step:** first run showed 7/21 — identical to Step 1's
*pre-fix* number, down to the same per-case breakdown and the same missed
quotes. That exact match was the signal something was wrong: traced it and
confirmed recency was scoring correctly (verified nonzero, varying bonus
values), it was the same round-robin merge bug silently discarding the
improvement. Fixed with the same RRF merge as Step 1. Corrected result:
**11/21 (52.4%)** — a real, distinct improvement over Step 1 alone, fixing
2 additional citations in the AMD Q3-earnings case.

### Parameter sweep — history_chunks 12 → 24

Reused Step 3's code completely unchanged; only doubled the retrieval budget.
**12/21 (57.1%)** — one more citation recovered.

**Diminishing returns, worth being explicit about:** the algorithm fix
(Steps 1+3) took retrieval from 19.0% to 52.4% — a 33.4-point gain. Doubling
the budget on top of that same algorithm added only 4.7 more points. Raising
`history_chunks` also isn't free in the real pipeline — it roughly doubles
the prior-side token count sent to the model on every call, a recurring cost
increase, not a one-time experiment cost. The algorithm change is the real
lever; budget is a minor, costed add-on.

## What remains unfixed — consistently, across every variant tried

**5 of 21 citations never get retrieved, under any combination tested:**

- A signature block ("...Corporate Vice President, Chief Accounting
  Officer...")
- An "emerging growth company" cover-page checkbox line
- A product-defect risk-factor sentence
- A safe-harbor forward-looking-statements disclaimer (2 occurrences)
- A GAAP-reconciliation-methodology footnote

All are near-universally duplicated legal/procedural boilerplate with no
real distinguishing content signal to rank on. This is treated as an
accepted, understood limitation, not a further target — no content-matching
approach can reasonably be expected to pick "the right" occurrence of a
sentence that's nearly identical everywhere it appears.

## Not yet done

- **Nothing here has been promoted into `net_new/`.** Every result above is
  an isolated, cheap, ground-truth-level check. The real validation step —
  fresh `--provider litellm` replay + `evaluation/grade.py` on the actual
  pipeline output, to see if better retrieval actually moves
  `recall_strict` in the full grader, not just this offline proxy — has not
  been run yet.
- Step 2 (IDF) was never re-tested combined with the fixed per-chunk + RRF
  merge — only tested against the original diluted query, where it clearly
  hurt. Whether IDF specifically fails *because* of query dilution, or fails
  regardless, is still an open question.
- The separately-proposed `current_char_budget` experiment (raising the
  *current*-document read limit, a different variable from everything in
  this file) has been explained but not executed.

## Cost

$0. Every step in this document is pure code against data already on disk —
no LLM calls, no judge, no replay.
