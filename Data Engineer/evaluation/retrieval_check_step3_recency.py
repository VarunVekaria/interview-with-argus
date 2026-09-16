"""Step 3 experiment: per-chunk queries (Step 1's proven win) + a recency
weighting layered on top of the existing overlap score. Deliberately does NOT
combine with Step 2's IDF weighting -- that made things worse (see
evaluation/results/retrieval-check-step2-idf.json), so it's dropped rather
than layered on top of something that already hurt.

What this targets: looking at what's STILL missed in Steps 0-2, a large
share are citations pointing at content that repeats near-identically across
many history chunks -- quarterly comparison tables (same header words every
quarter, only the numbers differ), standard disclaimers (safe-harbor,
GAAP-reconciliation boilerplate, risk-factor language). Multiple chunks score
almost the same by word overlap in these cases; picking the *right*
occurrence isn't a vocabulary problem, it's "which one is closest in time to
the filing being analyzed" -- exactly the signal available_at provides and
nothing else here uses.

Mechanism: keep raw overlap scoring and per-chunk querying unchanged;
multiply each chunk's score by (1 + freshness_bonus), where freshness_bonus
decays exponentially from 1.0 (same-day) toward 0 (a year+ old) based on
available_at. Deliberately mild and multiplicative on the EXISTING score, not
additive or a hard filter: a chunk with zero word overlap stays at zero
regardless of recency (recency only disambiguates among chunks that already
share real vocabulary; it never rescues total irrelevance), and a
strongly-matching older chunk can still beat a weakly-matching recent one.

Does not modify net_new/ -- validates the idea the same way Steps 1-2 were,
before promoting anything into the real pipeline.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

from net_new.dataset import Dataset
from net_new.parsing import Chunk, parse, tokens
from net_new.pipeline import config
from net_new.references import ReferenceCase, load_references

from evaluation.retrieval_check import locate_ground_truth_chunks
from evaluation.retrieval_check_step1_perchunk import selected_current_chunks

HALF_LIFE_DAYS = 180


def freshness_bonus(chunk_date: datetime, filing_date: datetime) -> float:
    days_ago = (filing_date - chunk_date).days
    if days_ago <= 0:
        return 1.0
    return math.exp(-days_ago / HALF_LIFE_DAYS)


def recency_retrieve(
    query: str,
    chunks: list[Chunk],
    limit: int,
    chunk_dates: dict[str, datetime],
    filing_date: datetime,
) -> list[Chunk]:
    wanted = tokens(query)

    def score(c: Chunk) -> float:
        overlap = len(wanted & tokens(c.text))
        if overlap == 0:
            return 0.0
        bonus = freshness_bonus(chunk_dates[c.document_id], filing_date)
        return overlap * (1 + bonus)

    ranked = sorted(chunks, key=lambda c: (-score(c), c.id))
    return [c for c in ranked if score(c) > 0][:limit]


def perchunk_recency_retrieve(
    current_chunks: list[Chunk],
    history_chunks: list[Chunk],
    overall_limit: int,
    chunk_dates: dict[str, datetime],
    filing_date: datetime,
) -> list[Chunk]:
    """Same Reciprocal Rank Fusion merge as Step 1 -- see that module's
    docstring for why round-robin-by-iteration-order was wrong. Fixed here
    too since this experiment builds directly on Step 1's merge."""
    rrf_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for current_chunk in current_chunks:
        if not current_chunk.text.strip():
            continue
        ranked = recency_retrieve(current_chunk.text, history_chunks, overall_limit, chunk_dates, filing_date)
        for rank, candidate in enumerate(ranked):
            rrf_scores[candidate.id] = rrf_scores.get(candidate.id, 0.0) + 1.0 / (rank + 1)
            chunk_by_id[candidate.id] = candidate
    ranked_ids = sorted(rrf_scores, key=lambda cid: (-rrf_scores[cid], cid))
    return [chunk_by_id[cid] for cid in ranked_ids[:overall_limit]]


def check_case(case: ReferenceCase, dataset: Dataset, settings: dict) -> list[dict]:
    filing = next(f for f in dataset.replay_filings() if f.filing_id == case.filing_id)
    history_docs = dataset.history(filing)
    history_chunks = [c for d in history_docs for c in parse(dataset, d)]
    chunk_dates = {d.id: d.available_at for d in history_docs}

    used = selected_current_chunks(dataset, filing, settings)
    retrieved_ids = {
        c.id
        for c in perchunk_recency_retrieve(
            used, history_chunks, settings["history_chunks"], chunk_dates, filing.available_at
        )
    }

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

    print("STEP 3 (per-chunk queries + recency-weighted scoring, half_life=180d)")
    print(f"Prior citations checked: {total}")
    print(f"  ground truth located: {located}/{total}")
    print(
        f"  retrieved (Step 3):    {retrieved}/{total}"
        + (f" ({retrieved / located:.1%} of located)" if located else "")
    )
    print("  retrieved (Step 0 baseline):           4/21 (19.0%)")
    print("  retrieved (Step 1, per-chunk query):    7/21 (33.3%)")
    print("  retrieved (Step 2, IDF, merged query):  3/21 (14.3%)")

    print("\n--- Per case (Step 3) ---")
    by_case: dict[str, list[dict]] = {}
    for row in all_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, rows in by_case.items():
        ret = sum(r["retrieved"] for r in rows)
        print(f"{case_id}: retrieved {ret}/{len(rows)}")
        for r in rows:
            if r["ground_truth_located"] and not r["retrieved"]:
                print(f"    still MISSED [{r['finding_id']}]: {r['quote'][:80]!r}")

    Path("evaluation/results/retrieval-check-step3-recency.json").write_text(
        json.dumps(
            {
                "method": "per-current-chunk queries + recency-weighted overlap scoring (half_life_days=180)",
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
    print("\nSaved: evaluation/results/retrieval-check-step3-recency.json")


if __name__ == "__main__":
    main()
