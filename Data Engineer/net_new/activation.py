"""Select a complete model run for the investor application."""

import json
from pathlib import Path

from .dataset import Dataset
from .pipeline import InvestorPrediction, load_records


def activate(dataset: Dataset, output: Path) -> None:
    output = output.resolve()
    try:
        relative = output.relative_to(dataset.root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("The active run must be inside the dataset directory") from exc
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if metadata.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Run does not match this dataset")
    if (
        metadata.get("status") != "completed"
        or metadata.get("config", {}).get("provider") not in {"openai", "litellm"}
    ):
        raise ValueError("Only a completed model run can be activated")
    records = load_records(output)
    expected = {f.filing_id: f.ticker for f in dataset.replay_filings()}
    if len(records) != len(expected) or {r["filing_id"] for r in records} != set(expected):
        raise ValueError("Run must cover every replay filing in this dataset")
    for record in records:
        if record["status"] != "completed" or record["ticker"] != expected[record["filing_id"]]:
            raise ValueError("Run contains failed or mismatched records")
        InvestorPrediction.model_validate(record["prediction"])
    path = dataset.root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["default_run"] = relative
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
