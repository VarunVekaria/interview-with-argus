from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def local_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    filing_id: str
    ticker: str
    company: str
    form: str
    role: Literal["primary", "exhibit"]
    available_at: datetime
    path: str
    source_url: str
    sha256: str

    @field_validator("available_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("available_at must include a timezone")
        return value.astimezone(UTC)


class Price(BaseModel):
    ticker: str
    date: date
    close: float = Field(gt=0, allow_inf_nan=False)
    session_close: datetime

    @field_validator("session_close")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("session_close must include a timezone")
        return value.astimezone(UTC)


class PriceInfo(BaseModel):
    path: str
    source: str
    source_url: str
    redistribution_basis: str
    adjustment: str
    currency: str
    kind: Literal["fictional", "market"]


class Manifest(BaseModel):
    schema_version: Literal[1]
    dataset_id: str
    title: str
    kind: Literal["fictional", "historical"]
    description: str
    replay_start: date
    replay_end: date
    documents: list[Document]
    prices: PriceInfo | None = None
    default_run: str | None = None

    @model_validator(mode="after")
    def valid_manifest(self) -> Manifest:
        if self.replay_end < self.replay_start:
            raise ValueError("replay_end precedes replay_start")
        if len({d.id for d in self.documents}) != len(self.documents):
            raise ValueError("Document IDs must be unique")
        if self.kind == "historical" and self.prices and self.prices.kind != "market":
            raise ValueError("Fictional prices cannot be attached to real companies")
        primaries = {d.filing_id: d for d in self.documents if d.role == "primary"}
        if len(primaries) != sum(d.role == "primary" for d in self.documents):
            raise ValueError("Each filing must have exactly one primary document")
        for doc in self.documents:
            primary = primaries.get(doc.filing_id)
            if not primary or (doc.ticker, doc.available_at) != (
                primary.ticker,
                primary.available_at,
            ):
                raise ValueError("Filing documents must share a primary, ticker and timestamp")
        return self


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = Manifest.model_validate_json(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.documents = sorted(self.manifest.documents, key=lambda d: (d.available_at, d.id))
        for doc in self.documents:
            actual = hashlib.sha256(local_file(self.root, doc.path).read_bytes()).hexdigest()
            if actual != doc.sha256:
                raise ValueError(f"Document checksum mismatch: {doc.id}")
        self.prices: list[Price] = []
        if self.manifest.prices:
            with local_file(self.root, self.manifest.prices.path).open(
                newline="", encoding="utf-8"
            ) as f:
                self.prices = [Price.model_validate(row) for row in csv.DictReader(f)]
            if len({(p.ticker, p.date) for p in self.prices}) != len(self.prices):
                raise ValueError("Duplicate ticker/session in prices")
            tickers = {d.ticker for d in self.documents}
            if any(p.ticker not in tickers for p in self.prices):
                raise ValueError("Prices contain a company absent from the corpus")
            self.prices.sort(key=lambda p: (p.ticker, p.date))
            for ticker in tickers:
                sessions = [p.session_close for p in self.prices if p.ticker == ticker]
                if sessions != sorted(set(sessions)):
                    raise ValueError("Session closes must increase with trading dates")

    @property
    def fingerprint(self) -> str:
        return digest(
            {
                "manifest": self.manifest.model_dump(mode="json", exclude={"default_run"}),
                "prices": [p.model_dump(mode="json") for p in self.prices],
            }
        )

    @property
    def tickers(self) -> list[str]:
        return sorted({d.ticker for d in self.documents})

    def replay_filings(self, ticker: str | None = None) -> list[Document]:
        from zoneinfo import ZoneInfo

        return [
            d
            for d in self.documents
            if d.role == "primary"
            and d.form in {"8-K", "8-K/A"}
            and (ticker is None or d.ticker == ticker)
            and self.manifest.replay_start
            <= d.available_at.astimezone(ZoneInfo("America/New_York")).date()
            <= self.manifest.replay_end
        ]

    def history(self, filing: Document) -> list[Document]:
        # Source history advances independently of model outputs or failures.
        return [
            d
            for d in self.documents
            if d.ticker == filing.ticker
            and d.available_at < filing.available_at
            and d.filing_id != filing.filing_id
        ]

    def current(self, filing: Document) -> list[Document]:
        return [d for d in self.documents if d.filing_id == filing.filing_id]

    def effective_session(self, filing: Document) -> date | None:
        # The first supplied trading close at/after release. No weekend/holiday guesses.
        return next(
            (
                p.date
                for p in self.prices
                if p.ticker == filing.ticker and p.session_close >= filing.available_at
            ),
            None,
        )

    def filings_for_window(self, ticker: str, start: date, end: date) -> list[Document]:
        if end < start:
            raise ValueError("Selection end precedes start")
        # This is a transparent temporal candidate rule, not causal attribution.
        return [
            d
            for d in self.replay_filings(ticker)
            if (session := self.effective_session(d)) is not None and start <= session <= end
        ]
