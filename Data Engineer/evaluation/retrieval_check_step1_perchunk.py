"""Step 1 experiment: per-current-chunk retrieval queries instead of one
merged blob for the whole filing. Scoring itself (retrieve()/tokens() in
net_new/parsing.py) is UNCHANGED -- this isolates the query-construction
variable from any scoring-algorithm change (that would be Step 2), per the
one-variable-at-a-time plan.

Baseline (Step 0, evaluation/retrieval_check.py): one query = every included
current chunk joined into a single string. A specific fact gets diluted by
unrelated text elsewhere in the same filing.

This variant: each included current chunk queries retrieve() independently.
Results are merged round-robin (best-of-each-chunk first, then
second-best-of-each, ...) and deduplicated, capped at the same overall
history_chunks budget (12) -- so the total amount of prior context reaching
the model stays comparable to baseline; only *how it's selected* changes.

Does not modify net_new/ -- validates the idea against the same ground-truth
check as Step 0 before promoting it into net_new/pipeline.py's build_input().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from net_new.dataset import Dataset
from net_new.parsing import Chunk, parse, retrieve
from net_new.pipeline import config
from net_new.references import ReferenceCase, load_references

from evaluation.retrieval_check import locate_ground_truth_chunks


def selected_current_chunks(dataset: Dataset, filing, settings: dict) -> list[Chunk]:
    """Same selection build_input() uses -- which current chunks fit the budget."""
    current_chunks = [c for d in dataset.current(filing) for c in parse(dataset, d)]
    used: list[Chunk] = []
    size = 0
    for chunk in current_chunks:
        if size + len(chunk.text) > settings["current_char_budget"]:
            break
        used.append(chunk)
        size += len(chunk.text)
    return used


def perchunk_retrieve(
    current_chunks: list[Chunk], history_chunks: list[Chunk], overall_limit: int
) -> list[Chunk]:
    per_chunk_results = [
        retrieve(c.text, history_chunks, overall_limit) for c in current_chunks if c.text.strip()
    ]
    merged: list[Chunk] = []
    seen: set[str] = set()
    rank = 0
    while len(merged) < overall_limit:
        added = False
        for lst in per_chunk_results:
            if rank < len(lst) and lst[rank].id not in seen:
                seen.add(lst[rank].id)
                merged.append(lst[rank])
                added = True
                if len(merged) >= overall_limit:
                    break
        if not added:
            break
        rank += 1
    return merged


def check_case(case: ReferenceCase, dataset: Dataset, settings: dict) -> list[dict]:
    filing = next(f for f in dataset.replay_filings() if f.filing_id == case.filing_id)
    history_chunks = [c for d in dataset.history(filing) for c in parse(dataset, d)]
    used = selected_current_chunks(dataset, filing, settings)
    retrieved_ids = {
        c.id for c in perchunk_retrieve(used, history_chunks, settings["history_chunks"])
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

    baseline_path = Path("evaluation/results/retrieval-check-baseline.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None

    print("STEP 1 (per-chunk queries, same scoring function) vs Step 0 baseline")
    print(f"Prior citations checked: {total}")
    print(f"  ground truth located: {located}/{total}")
    print(f"  retrieved (Step 1):     {retrieved}/{total}" + (f" ({retrieved / located:.1%} of located)" if located else ""))
    if baseline:
        b_ret, b_loc, b_total = baseline["retrieved"], baseline["ground_truth_located"], baseline["total_citations"]
        print(f"  retrieved (baseline):   {b_ret}/{b_total}" + (f" ({b_ret / b_loc:.1%} of located)" if b_loc else ""))

    print("\n--- Per case (Step 1) ---")
    by_case: dict[str, list[dict]] = {}
    for row in all_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, rows in by_case.items():
        ret = sum(r["retrieved"] for r in rows)
        print(f"{case_id}: retrieved {ret}/{len(rows)}")
        for r in rows:
            if r["ground_truth_located"] and not r["retrieved"]:
                print(f"    still MISSED [{r['finding_id']}]: {r['quote'][:80]!r}")

    Path("evaluation/results/retrieval-check-step1-perchunk.json").write_text(
        json.dumps(
            {
                "method": "per-current-chunk queries, unchanged retrieve()/tokens() scoring",
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
    print("\nSaved: evaluation/results/retrieval-check-step1-perchunk.json")


if __name__ == "__main__":
    main()
