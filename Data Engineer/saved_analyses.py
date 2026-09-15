"""Discover saved results using readable names and company-specific coverage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from net_new.dataset import Dataset, local_file
from net_new.pipeline import load_records


@dataclass
class SavedAnalysis:
    path: Path
    name: str
    metadata: dict
    records: dict[str, dict]
    preview: bool
    generated: str


def discover_analyses(dataset: Dataset, ticker: str, runs_root: Path) -> list[SavedAnalysis]:
    baseline = (
        local_file(dataset.root, dataset.manifest.default_run)
        if dataset.manifest.default_run
        else None
    )
    paths = ([baseline] if baseline else []) + [
        p.parent for p in sorted(runs_root.glob("*/run.json"), reverse=True)
    ]
    expected = {f.filing_id for f in dataset.replay_filings(ticker)}
    analyses = []
    for path in dict.fromkeys(paths):
        try:
            metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
            if metadata["dataset_fingerprint"] != dataset.fingerprint:
                continue
            records = {
                r["filing_id"]: r
                for r in load_records(path)
                if r["filing_id"] in expected and r["ticker"] == ticker
            }
            if not records:
                continue
            preview = metadata["config"]["provider"] == "extractive-preview"
            generated = (
                datetime.fromisoformat(metadata["started_at"])
                .astimezone(ZoneInfo("America/New_York"))
                .strftime("%b %d, %Y · %I:%M %p ET")
            )
            if path == baseline:
                name = "Supplied preview" if preview else "Supplied baseline"
            else:
                kind = "Source preview" if preview else "Model analysis"
                name = f"{kind} · {generated} · {metadata['run_id'][:6]}"
            analyses.append(SavedAnalysis(path, name, metadata, records, preview, generated))
        except (OSError, ValueError, KeyError, TypeError):
            # Partial writes and results from other workspaces are not selectable.
            continue
    return sorted(
        analyses, key=lambda a: (a.path != baseline, a.metadata["started_at"]), reverse=False
    )


def load_supplied_analysis(dataset: Dataset, ticker: str) -> SavedAnalysis | None:
    """The investor view always reads the dataset's supplied outputs."""
    if not dataset.manifest.default_run:
        return None
    path = local_file(dataset.root, dataset.manifest.default_run)
    metadata = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if metadata["dataset_fingerprint"] != dataset.fingerprint:
        raise ValueError("Supplied outputs do not match the dataset")
    expected = {filing.filing_id for filing in dataset.replay_filings(ticker)}
    records = {
        r["filing_id"]: r
        for r in load_records(path)
        if r["filing_id"] in expected and r["ticker"] == ticker
    }
    if metadata["config"]["provider"] not in {"openai", "litellm", "extractive-preview"}:
        raise ValueError("Authored references cannot be displayed as pipeline outputs")
    preview = metadata["config"]["provider"] == "extractive-preview"
    return SavedAnalysis(
        path,
        "Supplied preview" if preview else "Supplied baseline",
        metadata,
        records,
        preview,
        metadata["started_at"],
    )
