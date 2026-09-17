"""Real replay using the best-tested retrieval candidate from the
investigation in notes/retrieval-experiment.md: per-current-chunk queries +
recency-weighted scoring + Reciprocal Rank Fusion merge, history_chunks=24
(Step 3 + K24 there: 57.1% ground-truth hit rate vs. 19.0% baseline).

This is the first time that logic produces a REAL model run, not just the
offline ground-truth proxy check every other script in this investigation
used. Deliberately still does not modify net_new/ -- this mirrors
net_new.pipeline.build_input()/run_replay() closely enough to produce a
run.json + records.jsonl in the exact same shape (so evaluation/grade.py
works on it unmodified), but swaps only the retrieval step. The prompt,
schema, and model are completely unchanged from the real pipeline -- isolates
retrieval as the one variable, same discipline as every other step in this
investigation.

One deliberate exception: fresh_prediction_v2() below, not
net_new.pipeline.fresh_prediction(), does the actual model call. The largest
filing here (history_chunks=24 means up to double the prior-side content)
repeatedly failed against fresh_prediction()'s hardcoded timeout=90 and
max_output_tokens=6000, with no parameter to override either -- confirmed via
direct reproduction that failures were a mix of APITimeoutError (server-side
processing taking longer than 90s for the larger input) and ValidationError
(plausibly the response getting cut off before valid JSON completes: baseline
responses already used up to 5799 output tokens at the smaller 12-chunk
budget, so 6000 is a real, binding ceiling once the model has more retrieved
evidence to discuss, not just a timing issue). fresh_prediction_v2 raises both
ceilings and adds a small bounded retry for these specific, confirmed-transient
failure modes -- it does not change the prompt, schema, or model.

Spends real money: one real model call per replay filing, same as the
original baseline.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from net_new.dataset import Dataset, Document
from net_new.parsing import PARSER_VERSION, Chunk, parse
from net_new.pipeline import PROMPT, InvestorPrediction, digest, inference_connection

from evaluation.retrieval_check_step3_recency import perchunk_recency_retrieve


def fresh_prediction_v2(
    context: dict,
    settings: dict,
    *,
    timeout: float = 180,
    max_output_tokens: int = 8000,
    max_attempts: int = 3,
) -> tuple[InvestorPrediction, dict]:
    """Same call as net_new.pipeline.fresh_prediction, but with a longer
    timeout (90s -> 180s) and a larger max_output_tokens ceiling (6000 ->
    8000), plus a small bounded retry -- see the module docstring for why.
    Prompt, schema, and model are identical to the real pipeline."""
    from openai import OpenAI

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        client = OpenAI(**inference_connection(settings["provider"]), timeout=timeout, max_retries=0)
        try:
            raw_response = client.responses.with_raw_response.parse(
                model=settings["model"],
                instructions=PROMPT,
                input=json.dumps({"CURRENT": context["current"], "PRIOR": context["prior"]}),
                text_format=InvestorPrediction,
                max_output_tokens=max_output_tokens,
                store=False,
            )
            response = raw_response.parse()
            if response.output_parsed is None:
                raise ValueError(f"Model returned no structured prediction (status={response.status})")
            return response.output_parsed, {
                "response_id": response.id,
                "requested_model": settings["model"],
                "reported_model": response.model,
                "resolved_model": response.model if response.model != settings["model"] else None,
                "request_id": raw_response.headers.get("x-request-id"),
                "catalog_version": raw_response.headers.get("x-catalog-version"),
                "cost_usd": raw_response.headers.get("x-litellm-response-cost"),
                "usage": response.usage.model_dump() if response.usage else None,
                "raw_response": response.model_dump(mode="json", warnings=False),
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001 - bounded retry, confirmed-transient failure modes only
            last_exc = exc
    raise last_exc

RETRIEVAL_METHOD = (
    "per-current-chunk queries + recency-weighted overlap scoring "
    "(half_life_days=180) + Reciprocal Rank Fusion merge, history_chunks=24"
)


def build_input_v2(dataset: Dataset, filing: Document, settings: dict) -> dict:
    """Same shape as net_new.pipeline.build_input() -- only the prior-chunk
    selection differs (perchunk_recency_retrieve instead of retrieve())."""
    current_chunks = [c for d in dataset.current(filing) for c in parse(dataset, d)]
    all_history = dataset.history(filing)
    history_chunks = [c for d in all_history for c in parse(dataset, d)]
    used: list[Chunk] = []
    size = 0
    for chunk in current_chunks:
        if size + len(chunk.text) > settings["current_char_budget"]:
            break
        used.append(chunk)
        size += len(chunk.text)
    chunk_dates = {d.id: d.available_at for d in all_history}
    prior = perchunk_recency_retrieve(
        used, history_chunks, settings["history_chunks"], chunk_dates, filing.available_at
    )
    return {
        "filing_id": filing.filing_id,
        "ticker": filing.ticker,
        "available_at": filing.available_at.isoformat(),
        "current_document_ids": [d.id for d in dataset.current(filing)],
        "eligible_history_document_ids": [d.id for d in all_history],
        "current_chunks_total": len(current_chunks),
        "history_chunks_total": len(history_chunks),
        "current": [c.to_dict() for c in used],
        "prior": [c.to_dict() for c in prior],
    }


def run_replay_v2(dataset: Dataset, output: Path, model: str | None = None) -> dict:
    settings = {
        "pipeline_version": "net-new-v2-retrieval-experiment",
        "parser_version": PARSER_VERSION,
        "provider": "litellm",
        "model": model or os.getenv("ARGUS_MODEL", "anthropic-fast-v1"),
        "history_chunks": 24,
        "current_char_budget": 24000,
        "retrieval_method": RETRIEVAL_METHOD,
    }
    inference_connection("litellm")
    filings = dataset.replay_filings()
    if not filings:
        raise ValueError("No replay filings match this selection")
    if output.exists():
        raise ValueError("Output directory already exists; choose a new run directory")
    output.mkdir(parents=True)
    run_id = uuid4().hex
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_fingerprint": dataset.fingerprint,
        "execution_mode": "litellm",
        "config": settings,
        "prompt": PROMPT,
        "expected_filing_ids": [f.filing_id for f in filings],
        "status": "running",
        "note": "Experimental retrieval (see notes/retrieval-experiment.md); prompt/schema/model unchanged from net_new.pipeline, but calls go through fresh_prediction_v2 (timeout=180, max_output_tokens=8000, bounded retry) instead of fresh_prediction (timeout=90, max_output_tokens=6000) -- see module docstring.",
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    errors = 0
    try:
        with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
            for filing in filings:
                context = build_input_v2(dataset, filing, settings)
                input_hash = digest({"context": context, "config": settings})
                record = {
                    "run_id": run_id,
                    "filing_id": filing.filing_id,
                    "ticker": filing.ticker,
                    "available_at": filing.available_at.isoformat(),
                    "input_hash": input_hash,
                    "input": context,
                    "cache_hit": False,
                    "status": "completed",
                    "prediction": None,
                    "model_metadata": {},
                }
                started = time.perf_counter()
                try:
                    prediction, model_meta = fresh_prediction_v2(context, settings)
                    record["prediction"] = prediction.model_dump()
                    record["model_metadata"] = model_meta
                except Exception as exc:  # noqa: BLE001 - persist per-call failure and continue replay
                    record.update(status="error", error_type=type(exc).__name__)
                    record["error_message"] = "Inference failed; inspect error_type"
                record["generation_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
                errors += record["status"] != "completed"
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"  {filing.filing_id}: {record['status']}")
        metadata.update(status="completed_with_errors" if errors else "completed", errors=errors)
    except Exception:
        metadata.update(status="failed", errors=errors)
        raise
    finally:
        metadata["finished_at"] = datetime.now(UTC).isoformat()
        (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dataset = Dataset("data/pilot")
    output = Path("runs/retrieval-v2-replay")
    result = run_replay_v2(dataset, output)
    print(json.dumps({"status": result["status"], "errors": result["errors"], "output": str(output)}, indent=2))
