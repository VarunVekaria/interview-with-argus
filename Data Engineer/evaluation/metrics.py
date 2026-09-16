"""Aggregate matcher.py (semantic coverage) and evidence.py (citation grounding)
into the headline metrics: recall, classification agreement, evidence-validity
rate, and a supported/unsupported breakdown of extra findings.

Recall is reported as two numbers, deliberately, not one:
- recall_strict: reference findings fully captured (fact + the specific
  comparison it's testing for).
- recall_lenient: reference findings captured at all (fact stated, comparison
  may be missing).
On the baseline these are 69.6% and 95.7% — folding that into a single number
would hide exactly the failure this evaluation exists to surface (see
notes/case-review.md and evaluation/results/matcher-baseline.json).

Every rate is stored alongside its raw numerator/denominator so run-level
aggregation sums counts, never averages rates.

No model calls in this module — pure aggregation over matcher.py's saved
output and evidence.py's deterministic checks.
"""

from __future__ import annotations

from net_new.dataset import Dataset
from net_new.references import ReferenceCase

from .evidence import evidence_summary, verify_prediction_evidence


def case_metrics(case: ReferenceCase, matcher_row: dict, record: dict | None, dataset: Dataset) -> dict:
    base = {"case_id": case.case_id, "reference_findings_total": len(case.findings)}
    if matcher_row["output_state"] != "completed" or record is None:
        return {**base, "output_state": matcher_row["output_state"]}

    matches = matcher_row["matches"]
    total = len(matches)
    full = sum(1 for m in matches if m["coverage"] == "full")
    partial = sum(1 for m in matches if m["coverage"] == "partial")
    none = sum(1 for m in matches if m["coverage"] == "none")
    matched = full + partial
    classification_ok_matched = sum(
        1 for m in matches if m["coverage"] != "none" and m["classification_acceptable"]
    )

    evidence_reports = verify_prediction_evidence(dataset, record)
    ev_summary = evidence_summary(evidence_reports)

    extra_indices = matcher_row["extra_ai_indices"]
    extra_reports = [evidence_reports[i] for i in extra_indices]
    extra_verified = sum(1 for r in extra_reports if r.has_evidence and r.all_verified)
    extra_unverified = len(extra_indices) - extra_verified

    return {
        **base,
        "output_state": "completed",
        "coverage_full": full,
        "coverage_partial": partial,
        "coverage_none": none,
        "matched_count": matched,
        "recall_strict": full / total if total else None,
        "recall_lenient": matched / total if total else None,
        "classification_agreement_matched_count": classification_ok_matched,
        "classification_agreement_rate": (
            classification_ok_matched / matched if matched else None
        ),
        "ai_findings_total": matcher_row["ai_findings_total"],
        "extra_findings_total": len(extra_indices),
        "extra_findings_evidence_verified": extra_verified,
        "extra_findings_evidence_unverified": extra_unverified,
        "evidence": ev_summary,
        "missed_finding_ids": [m["finding_id"] for m in matches if m["coverage"] == "none"],
        "partial_finding_ids": [m["finding_id"] for m in matches if m["coverage"] == "partial"],
    }


def _sum(cases: list[dict], key: str) -> int:
    return sum(c.get(key) or 0 for c in cases)


def run_metrics(
    cases: list[ReferenceCase],
    matcher_results: list[dict],
    records: dict[str, dict],
    dataset: Dataset,
) -> dict:
    by_case = {c.case_id: c for c in cases}
    per_case = [
        case_metrics(by_case[row["case_id"]], row, records.get(by_case[row["case_id"]].filing_id), dataset)
        for row in matcher_results
    ]
    completed = [c for c in per_case if c["output_state"] == "completed"]

    total_ref = _sum(completed, "reference_findings_total")
    total_full = _sum(completed, "coverage_full")
    total_partial = _sum(completed, "coverage_partial")
    total_none = _sum(completed, "coverage_none")
    total_matched = _sum(completed, "matched_count")
    total_classification_ok = _sum(completed, "classification_agreement_matched_count")

    total_quotes = _sum([c["evidence"] for c in completed], "quotes_total")
    total_quotes_verified = _sum([c["evidence"] for c in completed], "quotes_verified")
    total_findings_no_evidence = _sum([c["evidence"] for c in completed], "findings_with_no_evidence")

    return {
        "cases_total": len(per_case),
        "cases_completed": len(completed),
        "cases_incomplete": [c["case_id"] for c in per_case if c["output_state"] != "completed"],
        "reference_findings_total": total_ref,
        "coverage_full": total_full,
        "coverage_partial": total_partial,
        "coverage_none": total_none,
        "recall_strict": total_full / total_ref if total_ref else None,
        "recall_lenient": total_matched / total_ref if total_ref else None,
        "classification_agreement_rate": (
            total_classification_ok / total_matched if total_matched else None
        ),
        "missed_finding_ids": [fid for c in completed for fid in c["missed_finding_ids"]],
        "ai_findings_total": _sum(completed, "ai_findings_total"),
        "extra_findings_total": _sum(completed, "extra_findings_total"),
        "extra_findings_evidence_verified": _sum(completed, "extra_findings_evidence_verified"),
        "extra_findings_evidence_unverified": _sum(completed, "extra_findings_evidence_unverified"),
        "evidence_quotes_total": total_quotes,
        "evidence_quotes_verified": total_quotes_verified,
        "evidence_quotes_verified_rate": (
            total_quotes_verified / total_quotes if total_quotes else None
        ),
        "findings_with_no_evidence_at_all": total_findings_no_evidence,
        "per_case": per_case,
    }
