"""Ground-truth diagnostic: does the current retrieve() actually surface the
specific prior chunk each reference finding's prior_evidence points to?

No model calls, no judgment calls -- this answers directly, from the reference
file's own citations, instead of inferring it from the model's self-reported
limitations. For each reference finding's prior_evidence quote:
  1. Find which history chunk(s) actually contain that quote (ground truth,
     located once against the full un-truncated history, independent of any
     retrieval logic).
  2. Run the real build_input()-equivalent query + retrieve() the pipeline
     would actually use for that filing.
  3. Check whether the ground-truth chunk made the retrieved top-K.

This is the "before" measurement for any retrieval-quality change. Re-run
this exact script after changing retrieve()/tokens() in net_new/parsing.py
and compare retrieved_rate -- that isolates whether a change actually
surfaces more of the right prior evidence, before spending anything on a
full replay + judge cycle to see if the model's output improves too.

Quote matching here uses plain whitespace normalization only (not the fuller
normalization in evidence.py) because reference quotes are already validated
byte-consistent (mod whitespace) with the source by references.load_references
itself -- see net_new/references.py's own quote check. There is nothing extra
to normalize for these specific quotes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from net_new.dataset import Dataset, Document
from net_new.parsing import Chunk, parse, retrieve
from net_new.pipeline import config
from net_new.references import ReferenceCase, load_references


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate_ground_truth_chunks(quote: str, chunks: list[Chunk]) -> list[str]:
    """chunk_id(s) whose text contains the quote. Normally one chunk; if the
    quote spans a chunk boundary, checks adjacent same-document chunk pairs."""
    q = _normalize(quote)
    hits = [c.id for c in chunks if q in _normalize(c.text)]
    if hits:
        return hits
    for a, b in zip(chunks, chunks[1:]):
        if a.document_id == b.document_id and q in _normalize(a.text + " " + b.text):
            return [a.id, b.id]
    return []


def build_query(dataset: Dataset, filing: Document, settings: dict) -> str:
    """Same current-chunk selection build_input() uses, so the query here is
    exactly what the real pipeline would have sent to retrieve()."""
    current_chunks = [c for d in dataset.current(filing) for c in parse(dataset, d)]
    used: list[Chunk] = []
    size = 0
    for chunk in current_chunks:
        if size + len(chunk.text) > settings["current_char_budget"]:
            break
        used.append(chunk)
        size += len(chunk.text)
    return " ".join(c.text for c in used)


def check_case(case: ReferenceCase, dataset: Dataset, settings: dict) -> list[dict]:
    filing = next(f for f in dataset.replay_filings() if f.filing_id == case.filing_id)
    history_chunks = [c for d in dataset.history(filing) for c in parse(dataset, d)]
    query = build_query(dataset, filing, settings)
    retrieved_ids = {c.id for c in retrieve(query, history_chunks, settings["history_chunks"])}

    rows = []
    for finding in case.findings:
        for citation in finding.prior_evidence:
            gt_ids = locate_ground_truth_chunks(citation.quote, history_chunks)
            rows.append(
                {
                    "case_id": case.case_id,
                    "finding_id": finding.finding_id,
                    "quote": citation.quote,
                    "ground_truth_chunk_ids": gt_ids,
                    "ground_truth_located": bool(gt_ids),
                    "retrieved": bool(gt_ids) and any(cid in retrieved_ids for cid in gt_ids),
                    "history_chunks_total": len(history_chunks),
                    "retrieved_count": len(retrieved_ids),
                }
            )
    return rows


def main() -> None:
    # Windows consoles default to cp1252; source quotes can contain characters
    # (curly quotes, bullets) that codec can't encode. Fix at the stream, once,
    # rather than special-casing every print() call below.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = Dataset("data/pilot")
    cases = load_references(Path("references/development.jsonl"), dataset)
    settings = config("extractive-preview")  # same current_char_budget/history_chunks, no API needed

    all_rows = [row for case in cases for row in check_case(case, dataset, settings)]

    total = len(all_rows)
    located = sum(r["ground_truth_located"] for r in all_rows)
    retrieved = sum(r["retrieved"] for r in all_rows)

    print(f"Prior citations checked: {total}")
    print(f"  ground truth chunk located (in un-truncated history): {located}/{total}")
    print(f"  ground truth chunk actually retrieved (top-{settings['history_chunks']}): {retrieved}/{total}")
    print(f"  retrieval hit rate (of located): {retrieved / located:.1%}" if located else "  n/a")
    print()
    print("--- Per case ---")
    by_case: dict[str, list[dict]] = {}
    for row in all_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, rows in by_case.items():
        loc = sum(r["ground_truth_located"] for r in rows)
        ret = sum(r["retrieved"] for r in rows)
        print(f"{case_id}: retrieved {ret}/{len(rows)} (located {loc}/{len(rows)})")
        for r in rows:
            if r["ground_truth_located"] and not r["retrieved"]:
                print(f"    MISSED [{r['finding_id']}]: {r['quote'][:90]!r}")
            elif not r["ground_truth_located"]:
                print(f"    NOT LOCATED (spans >2 chunks or boundary issue) [{r['finding_id']}]: {r['quote'][:90]!r}")

    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    Path("evaluation/results/retrieval-check-baseline.json").write_text(
        json.dumps(
            {
                "current_char_budget": settings["current_char_budget"],
                "history_chunks_limit": settings["history_chunks"],
                "total_citations": total,
                "ground_truth_located": located,
                "retrieved": retrieved,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nSaved: evaluation/results/retrieval-check-baseline.json")


if __name__ == "__main__":
    main()
