"""Align reference cases and pipeline outputs without prescribing metrics."""

import argparse
import json
from pathlib import Path

from .dataset import Dataset
from .pipeline import load_records
from .references import align_cases, load_references, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        dataset = Dataset(args.dataset)
        cases = load_references(args.references, dataset)
        metadata = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
        if metadata["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError("Run corpus fingerprint mismatch")
        aligned = align_cases(cases, load_records(args.run))
        if args.output.exists():
            raise ValueError("Output file exists; choose a new path")
        write_jsonl(args.output, aligned)
        print(
            json.dumps(
                {
                    "aligned_cases": len(aligned),
                    "missing_cases": sum(c["output_state"] == "missing" for c in aligned),
                    "failed_cases": sum(c["output_state"] == "error" for c in aligned),
                    "output": str(args.output),
                    "note": "Case alignment only; no accuracy score computed.",
                }
            )
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
