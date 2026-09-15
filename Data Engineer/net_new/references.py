"""Reference contracts and case alignment. Deliberately contains no scoring policy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field

from .dataset import Dataset, local_file


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    quote: str = Field(min_length=1)


class ReferenceFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding_id: str
    title: str
    acceptable_classifications: list[Literal["new", "changed", "repeated", "uncertain"]] = Field(
        min_length=1
    )
    expected_facts: list[str] = Field(min_length=1)
    comparison: str
    current_evidence: list[Quote] = Field(min_length=1)
    prior_evidence: list[Quote]
    acceptable_variations: list[str]
    adjudication_notes: str


class ReferenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_version: str
    case_id: str
    split: Literal["development", "holdout"]
    dataset_fingerprint: str
    filing_id: str
    ticker: str
    available_at: str
    current_document_ids: list[str]
    history_document_ids: list[str]
    title: str
    overview: str
    findings: list[ReferenceFinding]
    coverage: Literal["non_exhaustive"] = "non_exhaustive"
    other_supported_topics: list[str]
    limitations: list[str]
    review_status: Literal["source_checked_single_reviewer"] = "source_checked_single_reviewer"


def case_id(ticker: str, filing_id: str) -> str:
    return f"{ticker}:{filing_id}"


def normalized_source(dataset: Dataset, document_id: str) -> str:
    """Full original HTML text, independent of pipeline chunking/retrieval."""
    doc = next(d for d in dataset.documents if d.id == document_id)
    soup = BeautifulSoup(local_file(dataset.root, doc.path).read_bytes(), "html.parser")
    for node in soup(["script", "style", "ix:hidden", "noscript"]):
        node.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def load_references(path: Path, dataset: Dataset | None = None) -> list[ReferenceCase]:
    cases = [
        ReferenceCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate reference case IDs")
    if dataset is not None:
        docs = {}
        filings = {f.filing_id: f for f in dataset.replay_filings()}
        for c in cases:
            if c.dataset_fingerprint != dataset.fingerprint:
                raise ValueError("Reference corpus fingerprint mismatch")
            f = filings.get(c.filing_id)
            if f is None or c.case_id != case_id(f.ticker, f.filing_id) or c.ticker != f.ticker:
                raise ValueError("Reference case does not match replay filing")
            if c.available_at != f.available_at.isoformat():
                raise ValueError("Reference timestamp mismatch")
            current = {d.id for d in dataset.current(f)}
            history = {d.id for d in dataset.history(f)}
            if set(c.current_document_ids) != current or set(c.history_document_ids) != history:
                raise ValueError("Reference source coverage mismatch")
            if len({x.finding_id for x in c.findings}) != len(c.findings):
                raise ValueError("Duplicate reference finding IDs")
            for finding in c.findings:
                for field, eligible in [("current_evidence", current), ("prior_evidence", history)]:
                    for citation in getattr(finding, field):
                        if citation.document_id not in eligible:
                            raise ValueError("Reference citation outside point-in-time source set")
                        if citation.document_id not in docs:
                            docs[citation.document_id] = normalized_source(
                                dataset, citation.document_id
                            )
                        if (
                            re.sub(r"\s+", " ", citation.quote).strip()
                            not in docs[citation.document_id]
                        ):
                            raise ValueError("Reference quote not found in original document")
    return cases


def align_cases(cases: list[ReferenceCase], records: list[dict]) -> list[dict]:
    """Join whole cases, preserving errors/missing output. Candidates match findings."""
    by_id = {}
    expected = {c.case_id for c in cases}
    for record in records:
        key = case_id(record["ticker"], record["filing_id"])
        if key in by_id:
            raise ValueError(f"Duplicate output case: {key}")
        if key not in expected:
            raise ValueError(f"Output case absent from reference selection: {key}")
        by_id[key] = record
    return [
        {
            "case_id": c.case_id,
            "reference": c.model_dump(),
            "output_state": (
                "missing"
                if c.case_id not in by_id
                else "error"
                if by_id[c.case_id]["status"] != "completed"
                else "completed"
            ),
            "output": by_id.get(c.case_id),
        }
        for c in cases
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
