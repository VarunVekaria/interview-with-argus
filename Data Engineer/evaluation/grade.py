"""Grade a completed run against reference cases: judge semantic coverage
(matcher.py), verify evidence quotes (evidence.py via metrics.py), and write
the combined per-case and aggregate metrics to one file.

    uv run --env-file .env python -m evaluation.grade \\
        --dataset data/pilot --references references/development.jsonl \\
        --run data/pilot/baseline --output evaluation/results/baseline-graded.json

Spends real (small) money: one LLM-judge call per case in --run. Evidence
checking is free (deterministic, no model calls).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from net_new.dataset import Dataset
from net_new.pipeline import load_records
from net_new.references import load_references

from .matcher import match_run
from .metrics import run_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--judge-model",
        default="anthropic-strong-v1",
        help="Model alias for the LLM judge (see inference-status for options)",
    )
    args = parser.parse_args()
    try:
        dataset = Dataset(args.dataset)
        cases = load_references(args.references, dataset)
        run_metadata = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
        if run_metadata["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError("Run corpus fingerprint mismatch")
        records = load_records(args.run)
        records_by_filing = {r["filing_id"]: r for r in records}

        matcher_results, judge_meta = match_run(cases, records, args.judge_model)
        metrics = run_metrics(cases, matcher_results, records_by_filing, dataset)
        judge_cost = round(sum(float(m.get("cost_usd") or 0) for m in judge_meta), 5)

        if args.output.exists():
            raise ValueError("Output file exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "references": str(args.references),
                    "run": str(args.run),
                    "run_id": run_metadata.get("run_id"),
                    "run_config": run_metadata.get("config"),
                    "judge_model": args.judge_model,
                    "judge_cost_usd": judge_cost,
                    "metrics": metrics,
                    "matcher_results": matcher_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "cases_completed": metrics["cases_completed"],
                    "cases_incomplete": metrics["cases_incomplete"],
                    "recall_strict": metrics["recall_strict"],
                    "recall_lenient": metrics["recall_lenient"],
                    "classification_agreement_rate": metrics["classification_agreement_rate"],
                    "evidence_quotes_verified_rate": metrics["evidence_quotes_verified_rate"],
                    "judge_cost_usd": judge_cost,
                },
                indent=2,
            )
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
