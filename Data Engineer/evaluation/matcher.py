"""Match reference findings to AI findings for one case, via an LLM judge.

One judge call per case (not per finding-pair): the judge sees every reference
finding and every AI finding for that filing at once, and maps each reference
finding to the AI finding index/indices that cover it. This handles the
observed many-to-one splitting (one reference finding split into several AI
findings — see Case 1 in notes/case-review.md) without needing the candidate
to pre-guess a matching scheme, and keeps the judge's context coherent instead
of comparing isolated pairs.

Deliberately separate from evidence.py: this module judges *semantic*
coverage (does the AI's finding actually address the reference finding, and
does it capture the specific prior-period comparison the reference finding is
built on, not just the current-period number). Quote/citation grounding is a
fully deterministic, non-judgment check and stays in evidence.py.

"coverage" exists as its own field, separate from a plain matched/unmatched
bool, because the baseline shows a specific, recurring failure mode: the AI
states the current number correctly but never states the prior number needed
to make the actual comparison (see Case 3 in notes/case-review.md — ARR,
revenue-vs-guidance, and FY guidance all show this pattern). Collapsing that
into a single "matched" bool would hide the exact problem this evaluation
exists to surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from net_new.pipeline import inference_connection
from net_new.references import ReferenceCase, align_cases

JUDGE_PROMPT = """You are checking whether an AI pipeline's findings about a company
disclosure actually cover a set of expert-authored reference findings for the
same disclosure.

You will receive:
- REFERENCE_FINDINGS: reference findings, each with expected_facts (the facts
  that must be captured), a comparison (the specific prior-period comparison
  that makes this finding meaningful, if there is one), acceptable_classifications
  (labels that would be correct), acceptable_variations, and adjudication_notes.
- OTHER_SUPPORTED_TOPICS: topics the reference authors know are additionally
  valid but did not write up as full findings. An AI finding matching one of
  these is not unexpected or wrong; it simply will not match any
  REFERENCE_FINDINGS entry, which is fine.
- AI_FINDINGS: a 0-indexed list of findings the pipeline actually produced for
  this same disclosure, each with its own title, announced fact, change
  description, and classification.

Known pipeline behaviors to judge accordingly, not penalize by surprise:
- The pipeline sometimes splits one real fact into several separate AI
  findings. Multiple AI indices can jointly satisfy one reference finding.
- The pipeline sometimes reports a current-period number correctly without
  stating the specific prior-period figure needed to make the comparison the
  reference finding is testing for. That is a "partial" match, not "full",
  even when the current number is exactly right.

For EACH reference finding, decide:
- matched_ai_indices: the AI_FINDINGS indices (0-based) that together address
  it. Empty list if none do.
- coverage: "full" if the matched finding(s) state the SPECIFIC expected_facts
  (the actual required numbers/facts, not merely a related metric) AND the
  specific comparison described in `comparison`. "partial" if the specific
  expected_facts ARE stated but the comparison is missing, blurred, or wrong.
  "none" if the specific expected_facts are not stated at all — including when
  a thematically related but numerically different metric is discussed
  instead (e.g. an operating-level figure when expected_facts specifies a
  net/bottom-line figure is "none", not "partial": do not credit a different
  number as if it satisfied the required one).
- classification_acceptable: true only if at least one matched finding's
  classification is in this reference finding's acceptable_classifications.
  False if unmatched.
- rationale: one or two sentences, specific about what was or was not
  captured. Cite AI finding indices.

Do not reward vague topical overlap. A finding restating the current number
without the specific prior-period comparison the reference finding tests for
is "partial," never "full" — even when the current number is exactly right.
A finding that discusses a different, merely adjacent metric instead of the
one expected_facts actually specifies is "none," never "partial" — closeness
of topic is not closeness of fact.
"""


class FindingMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str
    matched_ai_indices: list[int]
    # rationale comes BEFORE the graded fields it justifies. Structured-output
    # generation follows field declaration order — putting coverage before
    # rationale let the model commit to a label before reasoning about it,
    # which produced a real observed self-contradiction (rationale concluding
    # "coverage is none" while the coverage field still said "partial").
    # Reasoning first, then the label it supports, avoids that.
    rationale: str
    coverage: Literal["full", "partial", "none"]
    classification_acceptable: bool


class MatcherOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    matches: list[FindingMatch]


def _reference_payload(case: ReferenceCase) -> list[dict]:
    return [
        {
            "finding_id": f.finding_id,
            "title": f.title,
            "expected_facts": f.expected_facts,
            "comparison": f.comparison,
            "acceptable_classifications": f.acceptable_classifications,
            "acceptable_variations": f.acceptable_variations,
            "adjudication_notes": f.adjudication_notes,
        }
        for f in case.findings
    ]


def _ai_payload(findings: list[dict]) -> list[dict]:
    return [
        {
            "index": i,
            "title": f["title"],
            "announced": f["announced"],
            "change": f["change"],
            "classification": f["classification"],
        }
        for i, f in enumerate(findings)
    ]


def judge_case(
    case: ReferenceCase, ai_findings: list[dict], model: str
) -> tuple[MatcherOutput, dict]:
    from openai import OpenAI

    client = OpenAI(**inference_connection("litellm"), timeout=90, max_retries=0)
    payload = {
        "REFERENCE_FINDINGS": _reference_payload(case),
        "OTHER_SUPPORTED_TOPICS": case.other_supported_topics,
        "AI_FINDINGS": _ai_payload(ai_findings),
    }
    raw_response = client.responses.with_raw_response.parse(
        model=model,
        instructions=JUDGE_PROMPT,
        input=json.dumps(payload),
        text_format=MatcherOutput,
        max_output_tokens=4000,
        store=False,
    )
    response = raw_response.parse()
    if response.output_parsed is None:
        raise ValueError(f"Judge returned no structured output (status={response.status})")
    meta = {
        "response_id": response.id,
        "model": model,
        "reported_model": response.model,
        "cost_usd": raw_response.headers.get("x-litellm-response-cost"),
        "usage": response.usage.model_dump() if response.usage else None,
    }
    result = response.output_parsed
    found_ids = {m.finding_id for m in result.matches}
    expected_ids = {f.finding_id for f in case.findings}
    if found_ids != expected_ids:
        raise ValueError(
            f"Judge output finding_ids {found_ids} do not match reference {expected_ids}"
        )
    n = len(ai_findings)
    for m in result.matches:
        bad = [i for i in m.matched_ai_indices if not (0 <= i < n)]
        if bad:
            raise ValueError(f"Judge cited out-of-range AI indices {bad} (only {n} findings)")
    return result, meta


def extra_ai_indices(ai_findings: list[dict], matches: list[FindingMatch]) -> list[int]:
    """AI finding indices that don't successfully cover any reference finding.

    Only counts matches with coverage "full" or "partial" as claiming an
    index. A "none"-coverage match can still list matched_ai_indices (the
    closest attempt the judge considered and rejected — useful context in the
    rationale) but that index has not actually covered anything, so it must
    still surface here rather than silently disappear because some reference
    finding's match object happened to name it.
    """
    matched = {i for m in matches if m.coverage != "none" for i in m.matched_ai_indices}
    return [i for i in range(len(ai_findings)) if i not in matched]


def match_run(
    cases: list[ReferenceCase], records: list[dict], model: str
) -> tuple[list[dict], list[dict]]:
    """Judge every completed case in a run. Returns (results, judge_call_metadata)."""
    aligned = align_cases(cases, records)
    by_case_id = {c.case_id: c for c in cases}
    results = []
    all_meta = []
    for row in aligned:
        case = by_case_id[row["case_id"]]
        if row["output_state"] != "completed":
            results.append({"case_id": row["case_id"], "output_state": row["output_state"]})
            continue
        ai_findings = row["output"]["prediction"]["findings"]
        matcher_output, meta = judge_case(case, ai_findings, model)
        all_meta.append({"case_id": row["case_id"], **meta})
        results.append(
            {
                "case_id": row["case_id"],
                "output_state": "completed",
                "matches": [m.model_dump() for m in matcher_output.matches],
                "extra_ai_indices": extra_ai_indices(ai_findings, matcher_output.matches),
                "ai_findings_total": len(ai_findings),
            }
        )
    return results, all_meta
