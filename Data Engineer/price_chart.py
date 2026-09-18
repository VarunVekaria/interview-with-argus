"""Local Plotly bridge: separate filing clicks from horizontal date selection."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import streamlit as st
from plotly.offline import get_plotlyjs
from streamlit.components.v1 import declare_component

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def chart_component():
    # Serve the installed Plotly library locally; no CDN or additional package is needed.
    runtime = ROOT / "artifacts" / "price-chart-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "index.html").write_text((ROOT / "ui/price_chart.html").read_text(encoding="utf-8"), encoding="utf-8")
    (runtime / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    return declare_component("price_history", path=runtime)


def event_date(value: str | float) -> date:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=UTC).date()
    return datetime.fromisoformat(value).date()


def resolve_selection(event: dict, filing_sessions: dict[str, date], bounds: tuple[date, date]):
    """Resolve UI events against known filings, preserving ranges with no matches."""
    if event.get("kind") == "filing":
        filing_id = event.get("filing_id")
        session = filing_sessions.get(filing_id)
        if session:
            return session, session, filing_id
    elif event.get("kind") == "range":
        try:
            start, end = sorted((event_date(event["start"]), event_date(event["end"])))
        except (ValueError, TypeError, KeyError, OverflowError, OSError):
            return None
        start, end = max(start, bounds[0]), min(end, bounds[1])
        if start <= end:
            return start, end, None
    return None
