
# Findings

## How I approached this

I started by reading the pipeline end to end before measuring anything how a
filing gets chunked, what counts as "current" versus "history", how prior context
is selected, and where the model call actually happens. That mattered later,
because the problem I eventually found wasn't in the model at all.

From there I worked outward in this order:

1. **Is the model making things up?** (evidence check)
2. **Is it finding the right facts?** (`recall_lenient`)
3. **Is it making the right comparisons?** (`recall_strict`)
4. **If not — why?** (retrieval investigation)

Each step only made sense after the one before it. If evidence had turned out to be
fabricated, there'd have been no point measuring recall at all.

## Step 1 — Is the evidence real, or is the model fabricating citations?

Before trusting any accuracy metric, I wanted to know whether the quotes the model
cites as evidence actually exist in the source documents, or whether it was inventing
them. This check needs no reference labels just the model's output and the real
filing so it was the cheapest place to start.

I checked all **140 evidence quotes** across the baseline's 74 findings against the
original source HTML.

| Result | Count |
|---|---|
| Quotes checked | 140 |
| Fabricated facts / invented citations | **0** |
| Failed a naive exact-substring check | 12 (26%) |
| Actually explained by formatting differences | 12 of 12 |

The important part: a naive check flags **26% as unverifiable**, which would look
alarming. I traced every one of them by hand, and none were fabrications. They were just model normalization and some formatting differences:


## Step 2 — Is it finding the right facts? (`recall_lenient`)

Next I compared the model's output against the labelled development references. I
split this into two separate measurements deliberately, because "getting it right"
means two different things here.

`recall_lenient` asks the simpler question: **does the model surface the fact the
reference says is there at all?**

**Result: 95.7% (22 of 23 reference findings).** Strong. The model is reliably
finding the substance of what's in these filings.

The one complete miss was `CRWD:0001535527-24-000024:f04` — the reference wanted
GAAP net loss of $16.8M and non-GAAP diluted EPS of $0.93; the model instead reported
GAAP *operating* loss of $55.7M and non-GAAP *operating* income of $194.9M. Related
metrics, but different line items.

## Step 3 — Is it making the right comparison? (`recall_strict`) — the real problem

`recall_strict` asks the harder question: did it capture the fact **and** the
specific prior-period comparison that makes it "net new"?

**Result: 69.6% (16 of 23).**

| Metric                                  | Result            |
| --------------------------------------- | ----------------- |
| `recall_lenient` (fact found)           | **95.7%** (22/23) |
| `recall_strict` (fact **+** comparison) | **69.6%** (16/23) |
| Gap                                     | **26 points**     |

That 26-point gap is the core finding of this work. The model almost always finds the
news, it frequently can't tell you what changed.

**The clearest example, CrowdStrike 26 Nov 2024:** the model correctly reports
`$153.0M net-new ARR` and frames it as a positive "$4 billion ARR milestone". What it
never says is that Q2 added **$217.6M** meaning this quarter is roughly a **30%
slowdown**, not a milestone. 

Crucially, the model told me why itself, in its own `limitations` field:

> "financial figures from Q2 FY2025… are not present in the retrieved prior excerpts."

It didn't invent the missing number it didn't *have* it. That pointed the
investigation away from the model and squarely at **retrieval**.

## Step 4 — Deep dive into retrieval

Reading `net_new/parsing.py` and `build_input()`, here's what retrieval actually does
currently:

- Take the whole current filing's included chunks and mash them into **one single
  query string**.
- Score every historical chunk by plain **word-overlap count** no weighting, no
  embeddings, no understanding.
- Pool every chunk from every prior filing together and take a **global top-12**.

Two structural problems fall out of that immediately. One merged query means a
specific fact's signal gets diluted by unrelated text from elsewhere in the same
filing. And plain word counting means two different dollar figures share zero
tokens a specific number can never itself drive a match.

**I measured this directly rather than assuming.** For each of the 21 `prior_evidence`
citations in the references, I checked whether the *exact chunk containing that quote*
made it into the model's retrieved set. 

**Baseline: 4 of 21 (19%).** Seven of nine cases retrieved *nothing* correct. The
evidence the model needed was sitting in history, findable, and never reached it.

## Step 5 — Four iterations on retrieval

| # | Change | Retrieved | Rate | Verdict |
|---|---|---|---|---|
| 0 | Baseline — one merged query, raw overlap, top-12 | 4 / 21 | 19.0% | — |
| 1 | **Per-current-chunk queries** (each chunk searches on its own) | 9 / 21 | **42.9%** | kept |
| 2 | **TF-IDF weighted scoring** | 3 / 21 | **14.3%** | **rejected — worse than baseline** |
| 3 | **Recency weighting** on top of per-chunk (180-day half-life) | 11 / 21 | **52.4%** | kept |
| 4 | Recency + **history chunks 12 → 24** | 12 / 21 | **57.1%** | kept |

**Overall: 19% → 57%, roughly tripling how often the right prior evidence reaches
the model.**

Two things worth being straight about, because they're not a clean upward story:

**Iteration 2 (TF-IDF) made things worse and I dropped it.** 


### Did better retrieval produce better answers?

I promoted the best configuration to a real model run and graded it identically.

| Metric                                     | Baseline | Improved retrieval | Δ         |
| ------------------------------------------ | -------- | ------------------ | --------- |
| `recall_strict`                            | 69.6%    | 69.6%              | **+-0.0** |
| `recall_lenient`                           | 95.7%    | 100.0%             | +4.3      |
| `classification_agreement`                 | 95.5%    | 91.3%              | −4.2      |
| Findings with **no prior evidence at all** | 64.9%    | **33.9%**          | **−31.0** |

Retrieval genuinely improved but `recall_strict` didnt move. 
The reason, being selection by the model. 

## Step 6 — The architecture change I'd make next

The failure mode that survives better retrieval is **comparison selection**, not
comparison availability.

Concretely (`AMD:0000002488-24-000161:f01`): both runs correctly state AMD's
data-centre revenue of **$3.5B**, and both make *a* comparison. But the improved-
retrieval run compares it **year-over-year**, while the reference is testing for the
**sequential** comparison against Q2's **$2.8B**. The model had the right material.
It picked a different, entirely legitimate comparison just not the one being asked
for.

More retrieved context didn't give it worse information. It gave it **more competing
choices about what to compare against**.

**So the change I'd make is architectural, not a tuning knob.** Today it's a single
prompt: here's the current filing, here are 24 prior chunks, go. Instead:

1. **Pass 1 — extract the facts.** Read only the current filing and enumerate the
   specific claims that need a comparison (each ARR figure, each guidance range, each
   margin).
2. **Pass 2 — retrieve per fact.** Run a targeted retrieval for *each* extracted fact
   independently, searching for that fact's prior-period counterpart specifically 
   not one pooled search for the whole document.
3. **Pass 3 — compose.** Write each finding against its own dedicated evidence.

This directly attacks both remaining problems: retrieval queries become narrow and
fact-specific instead of document-wide, and the model is no longer choosing which of
24 pooled chunks to talk about, because each finding arrives with its own matched
comparison. The cost is more model calls per filing, which needs measuring — the same
harness would measure it.

## Observations on the reference set

**It is explicitly non-exhaustive, and that materially limits what the metrics mean.**
The model produced **42 findings with no counterpart in the references**. I checked
them: **38 of 42 are fully evidence-grounded** in the source documents real,
supported content that simply isn't catalogued. (The other 4 are the same formatting
artifacts from Step 1, not fabrications.)

**One case worth raising with the reference authors.** In `AMD:0000002488-24-000161:f01`
the model made a valid year-over-year comparison, while the reference tests for the
sequential quarter-over-quarter one. Both are real, sourced comparisons of the same
metric; they differ only in *which prior period* is the reference point. I'm not
claiming the reference is wrong.

## Summary

| What I checked                                      | Result                               |
| --------------------------------------------------- | ------------------------------------ |
| Evidence fabrication                                | **0 / 140** — none                   |
| Finds the right facts (`recall_lenient`)            | **95.7%**                            |
| Makes the right comparison (`recall_strict`)        | **69.6%**                            |
| Correct classification labels                       | **95.5%**                            |
| Retrieval reaching the right prior evidence         | **19% → 57%** across four iterations |
| Findings left with no prior evidence at all         | **64.9% → 33.9%**                    |
| Net effect of improved retrieval on `recall_strict` | **flat** — the bottleneck moved      |

The pipeline is honest — it doesn't fabricate, and it says when it lacks context. It
reliably finds the news. Where it falls down is telling you what *changed*, and I
traced that to retrieval failing to surface the comparison figures 81% of the time.
Fixing retrieval tripled that hit rate, but the graded score stayed flat because the
bottleneck shifted from *having* the comparison to *choosing the right one* — which
is what the fact-level retrieval architecture above is designed to solve.


