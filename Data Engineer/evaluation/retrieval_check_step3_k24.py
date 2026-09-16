"""Parameter sweep on Step 3: identical per-chunk queries + recency-weighted
scoring + RRF merge, but history_chunks doubled from 12 to 24. Tests whether
simply retrieving more candidates helps on top of the algorithm fix,
independent of any further algorithm change.

Reuses evaluation.retrieval_check_step3_recency.check_case completely
unchanged -- only the settings dict passed into it differs. current_char_budget
is left at the pipeline default (24000); this isolates the history-side
retrieval budget as the only variable being tested here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from net_new.dataset import Dataset
from net_new.pipeline import config
from net_new.references import load_references

from evaluation.retrieval_check_step3_recency import check_case

NEW_HISTORY_CHUNKS = 24


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dataset = Dataset("data/pilot")
    cases = load_references(Path("references/development.jsonl"), dataset)
    settings = {**config("extractive-preview"), "history_chunks": NEW_HISTORY_CHUNKS}

    all_rows = [row for case in cases for row in check_case(case, dataset, settings)]
    total = len(all_rows)
    located = sum(r["ground_truth_located"] for r in all_rows)
    retrieved = sum(r["retrieved"] for r in all_rows)

    print(f"STEP 3 + history_chunks={NEW_HISTORY_CHUNKS} (pipeline default is 12)")
    print(f"Prior citations checked: {total}")
    print(f"  ground truth located: {located}/{total}")
    print(
        f"  retrieved:             {retrieved}/{total}"
        + (f" ({retrieved / located:.1%} of located)" if located else "")
    )
    print("  for comparison, all at history_chunks=12:")
    print("    Step 0 baseline:          4/21 (19.0%)")
    print("    Step 1 (per-chunk, RRF):  9/21 (42.9%)")
    print("    Step 3 (+ recency):      11/21 (52.4%)")

    print("\n--- Per case ---")
    by_case: dict[str, list[dict]] = {}
    for row in all_rows:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id, rows in by_case.items():
        ret = sum(r["retrieved"] for r in rows)
        print(f"{case_id}: retrieved {ret}/{len(rows)}")
        for r in rows:
            if r["ground_truth_located"] and not r["retrieved"]:
                print(f"    still MISSED [{r['finding_id']}]: {r['quote'][:80]!r}")

    Path("evaluation/results/retrieval-check-step3-k24.json").write_text(
        json.dumps(
            {
                "method": "Step 3 (per-chunk queries + recency + RRF merge) with history_chunks=24",
                "history_chunks_limit": NEW_HISTORY_CHUNKS,
                "total_citations": total,
                "ground_truth_located": located,
                "retrieved": retrieved,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nSaved: evaluation/results/retrieval-check-step3-k24.json")


if __name__ == "__main__":
    main()
