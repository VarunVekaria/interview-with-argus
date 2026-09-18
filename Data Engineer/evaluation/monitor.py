"""Label-free monitoring: health signals computable from a run's own records
alone, with no reference labels involved anywhere in this file. This is the
production-realistic half of the exercise -- once real filings start
arriving, there is no answer key.

These checks catch OPERATIONAL breakage (errors, emptiness, ungrounded
evidence, truncation, cost/latency blowups). They cannot tell you whether any
specific claim is factually correct, or whether a finding correctly captures
the prior-period comparison as opposed to just restating the current fact --
that distinction (this investigation's recall_strict vs recall_lenient gap)
needs a reference and a judge, and is structurally invisible to a label-free
monitor. See check_run()'s "invisible" section.

A classification-mix drift check was deliberately removed rather than kept as
a weak signal: its "normal" mix could only be established from a single
9-filing, 74-finding sample, it required 30+ findings before it would fire at
all, and it never triggered on any run tested. A threshold that thin is not
defensible as an alert -- better to ship five checks that are anchored to
something real than six where one is guesswork.

Every threshold below is anchored to the supplied baseline
(data/pilot/baseline, 9 filings, 0 errors) rather than invented -- each
constant says where its number came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from net_new.dataset import Dataset
from net_new.pipeline import load_records

from .evidence import evidence_summary, verify_prediction_evidence

# Established from the supplied baseline (data/pilot/baseline): 9 filings, 0 errors,
# 0 empty outputs, quotes_verified_rate 90%
# (evaluation/results/evidence-baseline-smoke.txt), worst single-filing truncation
# ratio 15/239 (AMD ZT Systems), worst latency 89014ms and worst cost $0.132
# (AMD:0000002488-24-000161).
BASELINE_EVIDENCE_VERIFIED_RATE = 0.90
BASELINE_WORST_TRUNCATION_RATIO = 15 / 239
BASELINE_WORST_LATENCY_MS = 89014
BASELINE_WORST_COST_USD = 0.132

# Thresholds: a margin beyond baseline's own worst historical observation, so a
# single benign outlier sitting exactly at that historical edge doesn't alert.
EVIDENCE_VERIFIED_RATE_FLOOR = 0.80
TRUNCATION_RATIO_FLOOR = 0.20
LATENCY_MS_CEILING = BASELINE_WORST_LATENCY_MS * 1.5
COST_USD_CEILING = BASELINE_WORST_COST_USD * 1.5


@dataclass
class Alert:
    check: str
    severity: str  # "error" | "warning"
    filing_id: str | None
    message: str


def check_errors(records: list[dict]) -> list[Alert]:
    return [
        Alert(
            "error_rate",
            "error",
            r["filing_id"],
            f"status={r['status']} error_type={r.get('error_type')} -- baseline established 0/9 errors",
        )
        for r in records
        if r["status"] != "completed"
    ]


def check_empty_output(records: list[dict]) -> list[Alert]:
    return [
        Alert(
            "empty_output",
            "warning",
            r["filing_id"],
            "0 findings returned -- baseline established 0/9 empty outputs",
        )
        for r in records
        if r["status"] == "completed" and len(r["prediction"]["findings"]) == 0
    ]


def check_evidence(dataset: Dataset, records: list[dict]) -> list[Alert]:
    alerts = []
    for r in records:
        if r["status"] != "completed":
            continue
        findings = verify_prediction_evidence(dataset, r)
        summary = evidence_summary(findings)
        rate = summary["quotes_verified_rate"]
        if rate is not None and rate < EVIDENCE_VERIFIED_RATE_FLOOR:
            alerts.append(
                Alert(
                    "evidence_validity",
                    "warning",
                    r["filing_id"],
                    f"quotes_verified_rate={rate:.1%} below floor {EVIDENCE_VERIFIED_RATE_FLOOR:.0%} "
                    f"(baseline established ~{BASELINE_EVIDENCE_VERIFIED_RATE:.0%})",
                )
            )
        # Hard trigger, independent of the rate: a newline in a quote is usually
        # the model's own legitimate bullet-join marker (evidence.py splits on it
        # and verifies each half now). It's only the corruption signature we
        # confirmed in the baseline when a segment STILL fails to verify even
        # after that split -- e.g. "def" standing in for an apostrophe, which has
        # no counterpart in the source at all. A bare newline is not the signal;
        # an unverified one is.
        for finding in findings:
            for c in finding.checks:
                if "\n" in c.quote and not c.verified:
                    alerts.append(
                        Alert(
                            "evidence_corruption",
                            "error",
                            r["filing_id"],
                            f"embedded newline in evidence quote (known decoding-corruption "
                            f"signature): {c.quote[:80]!r}",
                        )
                    )
    return alerts


def check_truncation(records: list[dict]) -> list[Alert]:
    alerts = []
    for r in records:
        if r["status"] != "completed":
            continue
        inp = r["input"]
        total = inp.get("current_chunks_total", 0)
        if total > 0:
            ratio = len(inp["current"]) / total
            if ratio < TRUNCATION_RATIO_FLOOR:
                alerts.append(
                    Alert(
                        "truncation",
                        "warning",
                        r["filing_id"],
                        f"only {ratio:.1%} of current document included "
                        f"({len(inp['current'])}/{total} chunks) -- below floor "
                        f"{TRUNCATION_RATIO_FLOOR:.0%} (worst baseline case was "
                        f"{BASELINE_WORST_TRUNCATION_RATIO:.1%})",
                    )
                )
    return alerts


def check_cost_latency(records: list[dict]) -> list[Alert]:
    alerts = []
    for r in records:
        if r["status"] != "completed":
            continue
        latency = r.get("generation_latency_ms")
        if latency and latency > LATENCY_MS_CEILING:
            alerts.append(
                Alert(
                    "latency",
                    "warning",
                    r["filing_id"],
                    f"generation_latency_ms={latency:.0f} exceeds ceiling "
                    f"{LATENCY_MS_CEILING:.0f} (1.5x worst baseline observation "
                    f"{BASELINE_WORST_LATENCY_MS})",
                )
            )
        cost = (r.get("model_metadata") or {}).get("cost_usd")
        try:
            cost = float(cost) if cost is not None else None
        except (TypeError, ValueError):
            cost = None
        if cost and cost > COST_USD_CEILING:
            alerts.append(
                Alert(
                    "cost",
                    "warning",
                    r["filing_id"],
                    f"cost_usd={cost:.4f} exceeds ceiling {COST_USD_CEILING:.4f} "
                    f"(1.5x worst baseline observation ${BASELINE_WORST_COST_USD})",
                )
            )
    return alerts


def prior_evidence_empty_summary(records: list[dict]) -> dict:
    """Informational, not a threshold alert: a finding CAN legitimately have no
    prior evidence (the prompt explicitly allows this for a genuinely new
    event). Reported as a trend to watch, not a pass/fail check."""
    total_findings = 0
    empty_prior = 0
    for r in records:
        if r["status"] == "completed":
            for f in r["prediction"]["findings"]:
                total_findings += 1
                if not f["prior_evidence"]:
                    empty_prior += 1
    return {
        "findings_total": total_findings,
        "findings_with_no_prior_evidence": empty_prior,
        "rate": (empty_prior / total_findings) if total_findings else None,
    }


INVISIBLE = [
    "Factual correctness of any specific claim -- these checks never compare against "
    "a reference or ground truth, only against the run's own output shape and the "
    "real source documents (for quote grounding).",
    "Whether a 'new'/'changed'/'repeated'/'uncertain' label is the CORRECT one for a "
    "given finding -- classification correctness needs the reference's "
    "acceptable_classifications, so it is graded, never monitored.",
    "Whether a finding captures the prior-period COMPARISON, as opposed to just "
    "restating the current fact -- the recall_strict vs recall_lenient gap this whole "
    "investigation is about requires a reference and a judge; it is structurally "
    "invisible to a label-free monitor.",
    "Whether an unmatched 'extra' finding is a genuinely good one -- evidence "
    "validity confirms the quote is real, not that the finding itself is worthwhile.",
]


def check_run(dataset: Dataset, records: list[dict]) -> dict:
    records = sorted(records, key=lambda r: r["available_at"])  # simulate arrival order
    alerts: list[Alert] = []
    alerts += check_errors(records)
    alerts += check_empty_output(records)
    alerts += check_evidence(dataset, records)
    alerts += check_truncation(records)
    alerts += check_cost_latency(records)

    return {
        "filings_total": len(records),
        "filings_completed": sum(r["status"] == "completed" for r in records),
        "alerts": [vars(a) for a in alerts],
        "alerts_by_severity": {
            sev: sum(1 for a in alerts if a.severity == sev) for sev in ("error", "warning")
        },
        "prior_evidence_empty": prior_evidence_empty_summary(records),
        "invisible": INVISIBLE,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        dataset = Dataset(args.dataset)
        records = load_records(args.run)
        result = check_run(dataset, records)
        if args.output.exists():
            raise ValueError("Output file exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"filings: {result['filings_completed']}/{result['filings_total']} completed")
        print(f"alerts by severity: {result['alerts_by_severity']}")
        if not result["alerts"]:
            print("No threshold crossed. Reporting that plainly -- not manufacturing an alert.")
        for a in result["alerts"]:
            print(f"  [{a['severity']}] {a['check']} {a['filing_id'] or ''}: {a['message']}")
        print(f"\nprior_evidence_empty: {result['prior_evidence_empty']}")
        print("\nWhat remains invisible to this monitor:")
        for note in result["invisible"]:
            print(f"  - {note}")
        print(f"\nSaved: {args.output}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
