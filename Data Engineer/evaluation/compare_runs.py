"""Compare two already-graded runs (evaluation/grade.py's output) -- per-case
and aggregate deltas, with findings that flip coverage state called out
explicitly. This is what "make changes measurable" means literally: point it
at a baseline and a candidate, get a reproducible diff instead of an ad hoc
one-off script.

Deliberately operates on grade.py's saved JSON output, not raw run
directories -- comparing is then free (pure JSON diffing, no re-judging, no
model calls).

Case-set mismatches are reported, never silently absorbed. A case that
completed in one run but errored/is missing in the other (exactly what
happened with AMD:0000002488-24-000121 in the retrieval-v2 experiment) is
excluded from the aggregate delta and listed separately -- otherwise the
aggregate comparison isn't apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _per_case_by_id(graded: dict) -> dict[str, dict]:
    return {c["case_id"]: c for c in graded["metrics"]["per_case"]}


def _matches_by_id(graded: dict) -> dict[str, dict[str, dict]]:
    """case_id -> {finding_id -> match dict} from matcher_results."""
    out = {}
    for row in graded["matcher_results"]:
        if row["output_state"] == "completed":
            out[row["case_id"]] = {m["finding_id"]: m for m in row["matches"]}
    return out


def _case_summary(c: dict) -> dict:
    return {
        "coverage_full": c["coverage_full"],
        "coverage_partial": c["coverage_partial"],
        "coverage_none": c["coverage_none"],
        "recall_strict": c["recall_strict"],
        "recall_lenient": c["recall_lenient"],
        "classification_agreement_rate": c["classification_agreement_rate"],
        "evidence_quotes_verified_rate": c["evidence"]["quotes_verified_rate"],
    }


def compare(baseline: dict, candidate: dict) -> dict:
    base_cases = _per_case_by_id(baseline)
    cand_cases = _per_case_by_id(candidate)
    base_matches = _matches_by_id(baseline)
    cand_matches = _matches_by_id(candidate)

    all_ids = set(base_cases) | set(cand_cases)
    both_completed, baseline_only, candidate_only = [], [], []
    per_case = []

    for case_id in sorted(all_ids):
        b = base_cases.get(case_id)
        c = cand_cases.get(case_id)
        b_ok = b is not None and b["output_state"] == "completed"
        c_ok = c is not None and c["output_state"] == "completed"

        if b_ok and c_ok:
            both_completed.append(case_id)
            flips = []
            for finding_id, bm in base_matches.get(case_id, {}).items():
                cm = cand_matches.get(case_id, {}).get(finding_id)
                if cm is None:
                    continue
                if bm["coverage"] != cm["coverage"] or bm["classification_acceptable"] != cm["classification_acceptable"]:
                    flips.append(
                        {
                            "finding_id": finding_id,
                            "baseline_coverage": bm["coverage"],
                            "candidate_coverage": cm["coverage"],
                            "baseline_classification_acceptable": bm["classification_acceptable"],
                            "candidate_classification_acceptable": cm["classification_acceptable"],
                        }
                    )
            per_case.append(
                {
                    "case_id": case_id,
                    "status": "both_completed",
                    "baseline": _case_summary(b),
                    "candidate": _case_summary(c),
                    "finding_flips": flips,
                }
            )
        elif b_ok:
            baseline_only.append(case_id)
            per_case.append(
                {
                    "case_id": case_id,
                    "status": "baseline_only_completed",
                    "baseline": _case_summary(b),
                    "candidate": None,
                    "candidate_output_state": c["output_state"] if c else "absent",
                    "finding_flips": [],
                }
            )
        elif c_ok:
            candidate_only.append(case_id)
            per_case.append(
                {
                    "case_id": case_id,
                    "status": "candidate_only_completed",
                    "baseline": None,
                    "baseline_output_state": b["output_state"] if b else "absent",
                    "candidate": _case_summary(c),
                    "finding_flips": [],
                }
            )
        # neither completed: not reportable, skip entirely

    def _agg(cases: dict, ids: list[str]) -> dict:
        full = sum(cases[i]["coverage_full"] for i in ids)
        partial = sum(cases[i]["coverage_partial"] for i in ids)
        none = sum(cases[i]["coverage_none"] for i in ids)
        total = full + partial + none
        matched = full + partial
        cls_ok = sum(
            round(cases[i]["classification_agreement_rate"] * (cases[i]["coverage_full"] + cases[i]["coverage_partial"]))
            for i in ids
            if cases[i]["classification_agreement_rate"] is not None
        )
        quotes_total = sum(cases[i]["evidence"]["quotes_total"] for i in ids)
        quotes_verified = sum(cases[i]["evidence"]["quotes_verified"] for i in ids)
        return {
            "cases": len(ids),
            "reference_findings_total": total,
            "coverage_full": full,
            "coverage_partial": partial,
            "coverage_none": none,
            "recall_strict": full / total if total else None,
            "recall_lenient": matched / total if total else None,
            "classification_agreement_rate": cls_ok / matched if matched else None,
            "evidence_quotes_verified_rate": quotes_verified / quotes_total if quotes_total else None,
        }

    base_agg = _agg(base_cases, both_completed)
    cand_agg = _agg(cand_cases, both_completed)

    def _delta(key: str) -> float | None:
        b, c = base_agg[key], cand_agg[key]
        return (c - b) if (b is not None and c is not None) else None

    return {
        "case_sets": {
            "both_completed": both_completed,
            "baseline_only_completed": baseline_only,
            "candidate_only_completed": candidate_only,
        },
        "aggregate": {
            "matched_cases": len(both_completed),
            "baseline": base_agg,
            "candidate": cand_agg,
            "delta": {
                "recall_strict": _delta("recall_strict"),
                "recall_lenient": _delta("recall_lenient"),
                "classification_agreement_rate": _delta("classification_agreement_rate"),
                "evidence_quotes_verified_rate": _delta("evidence_quotes_verified_rate"),
            },
        },
        "per_case": per_case,
    }


def _fmt_pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "n/a"


def print_summary(result: dict, baseline_path: str, candidate_path: str) -> None:
    cs = result["case_sets"]
    print(f"baseline:  {baseline_path}")
    print(f"candidate: {candidate_path}")
    print(f"\nmatched (completed in both): {len(cs['both_completed'])}")
    if cs["baseline_only_completed"]:
        print(f"  EXCLUDED, baseline-only-completed: {cs['baseline_only_completed']}")
    if cs["candidate_only_completed"]:
        print(f"  EXCLUDED, candidate-only-completed: {cs['candidate_only_completed']}")

    agg = result["aggregate"]
    print(f"\n--- aggregate, matched {agg['matched_cases']} cases only ---")
    for key in ["recall_strict", "recall_lenient", "classification_agreement_rate", "evidence_quotes_verified_rate"]:
        b, c, d = agg["baseline"][key], agg["candidate"][key], agg["delta"][key]
        arrow = "+" if (d or 0) >= 0 else ""
        print(f"  {key:32s} {_fmt_pct(b):>7s} -> {_fmt_pct(c):>7s}  ({arrow}{d:.1%})" if d is not None else f"  {key}: n/a")

    print("\n--- per case ---")
    for row in result["per_case"]:
        if row["status"] != "both_completed":
            print(f"  {row['case_id']}: {row['status']}")
            continue
        b, c = row["baseline"], row["candidate"]
        print(
            f"  {row['case_id']}: strict {_fmt_pct(b['recall_strict'])} -> {_fmt_pct(c['recall_strict'])}"
            + (f"   [{len(row['finding_flips'])} finding(s) flipped]" if row["finding_flips"] else "")
        )
        for flip in row["finding_flips"]:
            parts = []
            if flip["baseline_coverage"] != flip["candidate_coverage"]:
                parts.append(f"coverage {flip['baseline_coverage']} -> {flip['candidate_coverage']}")
            if flip["baseline_classification_acceptable"] != flip["candidate_classification_acceptable"]:
                parts.append(
                    f"classification_acceptable {flip['baseline_classification_acceptable']} -> "
                    f"{flip['candidate_classification_acceptable']}"
                )
            print(f"      {flip['finding_id']}: {', '.join(parts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Path to a grade.py output JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to a grade.py output JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
        result = compare(baseline, candidate)
        if args.output.exists():
            raise ValueError("Output file exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print_summary(result, str(args.baseline), str(args.candidate))
        print(f"\nSaved: {args.output}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
