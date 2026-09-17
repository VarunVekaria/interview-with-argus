"""Deterministic, code-only verification that a prediction's evidence quotes are real.

No model calls. This checks two independent things per citation:
  1. Was this document actually shown to the model for this call at all
     (i.e. present in input["current"] or input["prior"]), as opposed to
     merely "somewhere eligible"? A citation to unseen material is a
     distinct red flag from a wrong-but-real citation.
  2. Does the quote actually appear in the source — checked first against
     the SPECIFIC cited chunk (re-chunked from the original document), and
     only if that fails, against the whole document (a quote can legitimately
     span a chunk boundary near the 1600-char split point).

Quote comparison is normalized because raw exact-substring matching produces
a large false-fabrication rate that isn't real (verified by hand against
source HTML — see notes/case-review.md): the model normalizes typographic
quotes/apostrophes to ASCII, sometimes emits a stray backslash before an
apostrophe or percent sign, the HTML parser itself inserts spurious
whitespace when a tag interrupts a word, a quote spliced with "..." lowercases
the leading word of the resumed segment, and — seen when quoting slide-deck
source documents — a standalone " \\ " (backslash with spaces on both sides)
OR a literal embedded newline is the model's own marker for a line break
between two bullets that are not necessarily adjacent in the flattened source
text (confirmed by splitting on either and finding both resulting halves
verify independently against real source text). Both are treated as segment
boundaries, exactly like "...", not as content to normalize away.

A quote can still fail after all of this: a genuine decoding artifact was
found in the supplied baseline where a literal newline plus the word "def"
replaces an apostrophe (e.g. "Mr. Peng<newline>def s service"). Splitting on
the newline there does NOT rescue it — "def s service" still fails to verify
against the real source on its own, because "def" has no counterpart there at
all. That is the actual signature of fabricated/corrupted content: a segment
that still fails after every legitimate splice point has been split out.
Merely containing a newline or a backslash is not.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from net_new.dataset import Dataset
from net_new.parsing import parse
from net_new.references import normalized_source

_QUOTE_LIKE = re.compile(r"[‘’′“”″'\"]")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Comparison form: drop stray backslashes, remove all quote/apostrophe-like
    characters (curly or straight, single or double — removed, not unified, since
    a source ' vs model " mismatch is still the same underlying content), strip
    all whitespace, lowercase. See module docstring for why each step is needed."""
    text = text.replace("\\", "")
    text = _QUOTE_LIKE.sub("", text)
    return _WHITESPACE.sub("", text).lower()


_SEGMENT_BREAK = re.compile(r"\.\.\.|(?<=\s)\\(?=\s)|\n")


def _segments(quote: str) -> list[str]:
    """Split a spliced quote (on "..." or on a standalone " \\ ") into its
    independently-checkable parts. A stray backslash glued to punctuation with
    no surrounding whitespace (e.g. recipient\\'s) is not a separator and is
    left for _normalize() to strip instead."""
    return [p.strip() for p in _SEGMENT_BREAK.split(quote) if p.strip()] or [quote]


def _segments_found(quote: str, haystack_normalized: str) -> bool:
    return all(_normalize(seg) in haystack_normalized for seg in _segments(quote))


class QuoteCheck(BaseModel):
    document_id: str
    chunk_id: str
    quote: str
    role: Literal["current", "prior"]
    document_exists: bool
    in_shown_context: bool
    chunk_checked: bool  # False if chunk_id didn't resolve to a real chunk at all
    chunk_match: bool
    document_match: bool

    @property
    def verified(self) -> bool:
        return self.chunk_match or self.document_match


class FindingEvidence(BaseModel):
    finding_index: int
    title: str
    classification: str
    checks: list[QuoteCheck]

    @property
    def has_evidence(self) -> bool:
        return len(self.checks) > 0

    @property
    def all_verified(self) -> bool:
        return all(c.verified for c in self.checks)

    @property
    def all_in_shown_context(self) -> bool:
        return all(c.in_shown_context for c in self.checks)


_doc_index_cache: dict[int, dict[str, object]] = {}
_chunk_cache: dict[tuple[int, str], dict[str, str]] = {}
_source_cache: dict[tuple[int, str], str] = {}


def _document_by_id(dataset: Dataset, document_id: str):
    key = id(dataset)
    index = _doc_index_cache.setdefault(key, {d.id: d for d in dataset.documents})
    return index.get(document_id)


def _chunks_for_document(dataset: Dataset, document_id: str) -> dict[str, str]:
    key = (id(dataset), document_id)
    if key not in _chunk_cache:
        doc = _document_by_id(dataset, document_id)
        _chunk_cache[key] = {c.id: c.text for c in parse(dataset, doc)} if doc else {}
    return _chunk_cache[key]


def _source_text(dataset: Dataset, document_id: str) -> str:
    key = (id(dataset), document_id)
    if key not in _source_cache:
        _source_cache[key] = normalized_source(dataset, document_id)
    return _source_cache[key]


def shown_context_ids(record: dict) -> tuple[set[str], set[str]]:
    """Document IDs actually shown to the model for this call — not the whole
    eligible history, only what was in input["current"]/input["prior"]."""
    inp = record["input"]
    current_ids = {c["document_id"] for c in inp["current"]}
    prior_ids = {c["document_id"] for c in inp["prior"]}
    return current_ids, prior_ids


def verify_quote(
    dataset: Dataset,
    document_id: str,
    chunk_id: str,
    quote: str,
    role: Literal["current", "prior"],
    shown_ids: set[str],
) -> QuoteCheck:
    doc = _document_by_id(dataset, document_id)
    if doc is None:
        return QuoteCheck(
            document_id=document_id,
            chunk_id=chunk_id,
            quote=quote,
            role=role,
            document_exists=False,
            in_shown_context=False,
            chunk_checked=False,
            chunk_match=False,
            document_match=False,
        )
    chunks = _chunks_for_document(dataset, document_id)
    chunk_text = chunks.get(chunk_id)
    chunk_checked = chunk_text is not None
    chunk_match = chunk_checked and _segments_found(quote, _normalize(chunk_text))
    document_match = _segments_found(quote, _normalize(_source_text(dataset, document_id)))
    return QuoteCheck(
        document_id=document_id,
        chunk_id=chunk_id,
        quote=quote,
        role=role,
        document_exists=True,
        in_shown_context=document_id in shown_ids,
        chunk_checked=chunk_checked,
        chunk_match=chunk_match,
        document_match=document_match,
    )


def verify_finding_evidence(dataset: Dataset, record: dict, finding: dict, index: int) -> FindingEvidence:
    current_shown, prior_shown = shown_context_ids(record)
    checks = [
        verify_quote(dataset, e["document_id"], e["chunk_id"], e["quote"], "current", current_shown)
        for e in finding["current_evidence"]
    ] + [
        verify_quote(dataset, e["document_id"], e["chunk_id"], e["quote"], "prior", prior_shown)
        for e in finding["prior_evidence"]
    ]
    return FindingEvidence(
        finding_index=index,
        title=finding["title"],
        classification=finding["classification"],
        checks=checks,
    )


def verify_prediction_evidence(dataset: Dataset, record: dict) -> list[FindingEvidence]:
    if record["status"] != "completed" or not record.get("prediction"):
        return []
    findings = record["prediction"]["findings"]
    return [verify_finding_evidence(dataset, record, f, i) for i, f in enumerate(findings)]


def evidence_summary(all_findings: list[FindingEvidence]) -> dict:
    all_checks = [c for f in all_findings for c in f.checks]
    n_checks = len(all_checks)
    return {
        "findings_total": len(all_findings),
        "findings_with_no_evidence": sum(1 for f in all_findings if not f.has_evidence),
        "findings_fully_verified": sum(1 for f in all_findings if f.has_evidence and f.all_verified),
        "quotes_total": n_checks,
        "quotes_verified": sum(c.verified for c in all_checks),
        "quotes_verified_rate": (sum(c.verified for c in all_checks) / n_checks) if n_checks else None,
        "quotes_verified_by_chunk": sum(c.chunk_match for c in all_checks),
        "quotes_verified_by_document_fallback_only": sum(
            c.verified and not c.chunk_match for c in all_checks
        ),
        "quotes_outside_shown_context": sum(not c.in_shown_context for c in all_checks),
        "quotes_document_missing": sum(not c.document_exists for c in all_checks),
    }
