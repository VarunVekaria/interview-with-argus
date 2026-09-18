"""Step 2 experiment: IDF-weighted overlap scoring instead of raw word-overlap
count. Query construction is UNCHANGED (back to Step 0's single merged query
per filing, not Step 1's per-chunk split) -- isolates the scoring variable
alone, same one-variable-at-a-time approach used for Step 1.

Problem this targets: retrieve()'s current score is len(wanted & chunk_tokens)
-- every shared word counts equally. A boilerplate word appearing in nearly
every history chunk ("company", "quarter", "accordingly") contributes the same
as a rare, genuinely on-topic word. Two different dollar figures never share
a token at all, so a specific number can never itself drive a match; only
surrounding vocabulary can, and that vocabulary is exactly what's getting
drowned out by common words.

Fix: weight each shared word by its inverse document frequency across THAT
case's own history corpus -- a word appearing in only a few chunks scores much
higher than one appearing in most of them. This is a simplified IDF-weighted
proxy, not textbook BM25 (no term-frequency or document-length normalization
component) -- chunks are a fixed ~1600 chars, so within-chunk term frequency
adds little here; the IDF half is what actually targets the boilerplate
problem, so that's what's implemented, deliberately kept simple to explain
and verify.

Does not modify net_new/ -- validates the idea the same way Step 1 was
validated, before promoting anything into the real pipeline.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

from net_new.dataset import Dataset
from net_new.parsing import Chunk, parse, tokens
from net_new.pipeline import config
from net_new.references import ReferenceCase, load_references

from evaluation.retrieval_check import build_query, locate_ground_truth_chunks


def document_frequencies(chunks: list[Chunk]) -> Counter[str]:
    df: Counter[str] = Counter()
    for c in chunks:
        df.update(tokens(c.text))
    return df


def idf_weights(df: Counter[str], n_docs: int) -> dict[str, float]:
    # Smoothed IDF: log((N+1)/(df+1)) + 1, always positive, never divides by zero.
    return {word: math.log((n_docs + 1) / (count + 1)) + 1 for word, count in df.items()}


def idf_retrieve(
    query: str, chunks: list[Chunk], limit: int, idf: dict[str, float]
) -> list[Chunk]:
    wanted = tokens(query)

    def score(c: Chunk) -> float:
        return sum(idf.get(w, 0.0) for w in wanted & tokens(c.text))

    ranked = sorted(chunks, key=lambda c: (-score(c), c.id))
    return [c for c in ranked if score(c) > 0][:limit]


def check_case(case: ReferenceCase, dataset: Dataset, settings: dict) -> list[dict]:
    filing = next(f for f in dataset.replay_filings() if f.filing_id == case.filing_id)
    history_chunks = [c for d in dataset.history(filing) for c in parse(dataset, d)]
    query = build_query(dataset, filing, settings)

    df = document_frequencies(history_chunks)
    idf = idf_weights(df, len(history_chunks))
    retrieved_ids = {c.id for c in idf_retrieve(query, history_chunks, settings["history_chunks"], idf)}

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
                }
            )
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dataset = Dataset("data/pilot")
    cases = load_references(Path("references/development.jsonl"), dataset)
    settings = config("extractive-preview")

    all_rows = [row for case in cases for row in check_case(case, dataset, settings)]
    total = len(all_rows)
    located = sum(r["ground_truth_located"] for r in all_rows)
    retrieved = sum(r["retrieved"] for r in all_rows)

    print("STEP 2 (IDF-weighted scoring, original single merged-query construction)")
    print(f"Prior citations checked: {total}")
    print(f"  ground truth located: {located}/{total}")
    print(f"  retrieved (Step 2):   {retrieved}/{total}" + (f" ({retrieved / located:.1%} of located)" if located else ""))
    print("  retrieved (Step 0 baseline): 4/21 (19.0%)")
    print("  retrieved (Step 1, per-chunk query): 7/21 (33.3%)")

    print("\n--- Per case (Step 2) ---")
    by_case: dict[str, list[dict]] = {}
    for row in all_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, rows in by_case.items():
        ret = sum(r["retrieved"] for r in rows)
        print(f"{case_id}: retrieved {ret}/{len(rows)}")
        for r in rows:
            if r["ground_truth_located"] and not r["retrieved"]:
                print(f"    still MISSED [{r['finding_id']}]: {r['quote'][:80]!r}")

    Path("evaluation/results/retrieval-check-step2-idf.json").write_text(
        json.dumps(
            {
                "method": "IDF-weighted overlap scoring, original single merged-query construction",
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
    print("\nSaved: evaluation/results/retrieval-check-step2-idf.json")


if __name__ == "__main__":
    main()
