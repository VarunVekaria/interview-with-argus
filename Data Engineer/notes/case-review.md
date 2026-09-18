# Case review notes (Phase 1)

Done via direct comparison of `references/development.jsonl` against
`data/pilot/baseline/records.jsonl` (not the browser UI) — see chat for the
scripts. Every evidence quote below was checked against the real source HTML,
not just eyeballed.

## Case 1 — AMD · Jul 02, 2024 (executive salary increases)

- AI finding count: 6. Reference finding count: 2.
- Coverage: AI findings 1-3 (Su/Hu/Guido+Norrod salaries) together = reference f01. AI findings 4-6
  (LTI target values, PRSU mechanics, vesting schedule) together = reference f02 plus its
  explicitly-anticipated extra topic ("detailed performance vesting and payout conditions").
  Every dollar figure matches the reference exactly.
- Classification: AI used "new" for all 6. Reference allows ["new","changed"] for both — "new" is
  defensible since the AI's own stated limitation says no prior comp filing was retrieved.
- Evidence: all 6 findings' quotes are real once you normalize typographic punctuation (see
  "methodology note" below). Zero fabrication.
- **Verdict: substantively correct, just over-fragmented.** 3x more findings than the reference for
  the same facts. This is a matching-granularity problem for the grader, not an accuracy problem.

## Case 2 — AMD · Aug 19, 2024 (ZT Systems) — only 15/239 chunks (6%) of this filing were read

- AI finding count: 8. Reference finding count: 2.
- Coverage: AI finding 1 = reference f01's core fact ($4.9B deal). AI finding 5 = reference f02
  (manufacturing separation). AI findings 2, 3, 4 = the reference's own "other_supported_topics"
  list ($3.375B cash structure, contingent consideration, $300M termination fee) — the reference
  explicitly flagged these as valid-but-optional, and the AI found them correctly.
- AI findings 6 and 7 (key-employee retention terms, S-3 resale registration) are genuinely extra —
  not in the reference or its optional-topics list. Verified: both are real, sourced correctly.
- AI finding 8 correctly identifies an unrelated item (Victor Peng's retirement) bundled in the same
  8-K and correctly classifies it "repeated," citing the July 20 filing where it first appeared.
- One soft miss: reference f01 mentions "closing expected in the first half of 2025" — I could not
  find this specific expectation stated anywhere in the AI's 8 findings, though the Aug 17, 2025
  regulatory deadline it does cite is adjacent. Possibly one of the 224 dropped chunks.
- **Verdict: despite reading only 6% of the document, this case is not obviously harmed.** The 8-K's
  primary text (the announcement itself) apparently front-loads the key facts before the huge
  attached merger-agreement exhibit that inflates the chunk count — so high truncation ratio alone
  doesn't prove damage. Worth checking other filing types (10-K/10-Q) where key figures sit
  mid-document instead, before generalizing this "truncation is fine" reading.

## Case 3 — CRWD · Nov 26, 2024 (Q3 earnings) — the most revealing case

- AI finding count: 13. Reference finding count: 4.
- The AI gets the **current-quarter numbers right for 3 of 4** reference findings (ARR $4.02B/+27%,
  revenue $1.01B/+29%, FY25 guidance $3,923.8-3,930.5M all match exactly) — but **misses the
  comparison that is the actual point of each reference finding**:
  - f03 wants "net-new ARR fell to $153.0M from Q2's $217.6M, a ~30% sequential decline." The AI
    reports the $153.0M figure but never mentions Q2's $217.6M or the deceleration — it frames the
    same number as a positive "first time crossing $4B" milestone instead.
  - f02 wants "revenue beat August's $979.2-984.7M guidance ceiling by $25.5M." The AI reports the
    $1.01B figure but never states the prior guidance range it beat.
  - f01 wants the two-sided comparison (up from August's cut, still below June's original). The AI
    reports the new range correctly but not the comparison.
  - **This is not a coincidence** — the AI's own stated limitation says: "financial figures from Q2
    FY2025 results... are not present in the retrieved prior excerpts." Its retrieval never
    surfaced the comparison numbers, and it told you so honestly instead of inventing them.
- **f04 (GAAP net loss $16.8M vs. non-GAAP diluted EPS $0.93) appears to be missed entirely.** The
  AI instead reports adjacent but different figures (GAAP *operating* loss $55.7M, non-GAAP
  *operating* income $194.9M) — never the specific net-loss-vs-non-GAAP-EPS contrast the reference
  is built on. This is a genuine, clean miss, not a granularity quibble.
- Classification: used "changed" 3x and "repeated" 1x here (unlike cases 1-2, all "new"). The
  "changed" uses are reasonable (e.g. correctly noticing an "eight-or-more-module" disclosure tier
  that didn't exist in the prior filing). The "repeated" use (July 19 Incident risk factor) is
  correct.
- **Verdict: the pipeline can find and state current facts reliably, but its retrieval step is the
  bottleneck for the actual "net new" comparison** — it often can't locate the specific prior-period
  number to compare against, so it reports news without being able to say confidently whether it's
  better, worse, or unchanged. That's the core product problem in one sentence.

## Methodology note: evidence-quote verification is noisier than a plain substring check

I checked all 46 evidence quotes across these 3 cases against the real source HTML. A naive
"is this quote an exact substring of the source" check fails on **12 of 46 (26%)** — but every
single failure turned out to be one of these four artifacts, not fabrication:
1. Source uses typographic quotes/apostrophes (’ “ ”); the model normalizes them to straight ASCII.
2. The model sometimes emits a literal stray backslash before apostrophes/percent signs (e.g.
   `recipient\'s`, `250 %\`) — an LLM output quirk, not a content error.
3. The HTML parser itself inserts spurious whitespace when tags interrupt a word or a footnote
   marker sits inline (e.g. real source text is "Equivalen ts", not "Equivalents").
4. When the model splices two non-adjacent passages with "...", the leading word of the second
   segment gets lowercased (stylistic), which fails a case-sensitive check against the source's
   sentence-initial capital.
- After normalizing for all four, **0 of 46 quotes were unverifiable** in these 3 cases.
- **Takeaway for the grader:** an evidence-validity checker needs quote-character normalization,
  backslash-stripping, ellipsis-aware per-segment matching, and case-insensitive comparison — or
  it will report a false ~26% fabrication rate that isn't real. This is going in FINDINGS.md as a
  design decision, with this exact evidence.

## Overall impression after all 3

- **Biggest single problem:** not classification labels, not fabricated evidence (there wasn't
  any) — it's that lexical/token-overlap retrieval frequently fails to surface the *specific prior
  number* needed to make the comparison that "net new" is supposed to be about. The model then
  either omits the comparison (Case 3) or over-fragments one fact into many findings (Case 1)
  rather than saying something false.
- **What it did well:** every quote checked was genuinely grounded in a real document once
  normalized. It correctly used "repeated" twice (both times correctly). It self-reported its own
  retrieval gaps in the `limitations` field rather than hallucinating past them.
