from __future__ import annotations

import os
import re
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import streamlit as st

from net_new.dataset import Dataset
from net_new.parsing import parse
from price_chart import chart_component, resolve_selection
from saved_analyses import load_supplied_analysis

ROOT = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
st.set_page_config(page_title="Disclosure research · Argus", layout="wide")
st.html(f"<style>{(ROOT / 'ui.css').read_text(encoding='utf-8')}</style>")


def html(value: str) -> None:
    st.html(value)


def safe(value: object) -> str:
    return escape(str(value))


@st.cache_data(show_spinner=False)
def source_text(root: str, document_id: str, checksum: str) -> str:
    source = Dataset(root)
    doc = next(d for d in source.documents if d.id == document_id)
    return "\n\n".join(c.text for c in parse(source, doc))


def text_for(doc) -> str:
    return source_text(str(dataset.root), doc.id, doc.sha256)


ITEMS = {
    "1.01": "Material agreement",
    "1.02": "Agreement terminated",
    "2.01": "Acquisition or disposition",
    "2.02": "Financial results",
    "2.05": "Restructuring update",
    "5.02": "Leadership & compensation",
    "5.07": "Shareholder vote",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other events",
}


def filing_title(doc) -> str:
    record = records.get(doc.filing_id, {})
    if headline := (record.get("prediction") or {}).get("headline") or record.get("headline"):
        return headline
    body = text_for(doc)
    codes = re.findall(r"\bItem\s+(\d\.\d{2})\b", body, flags=re.IGNORECASE)
    return next((ITEMS[code] for code in codes if code in ITEMS), "Company update")


def source_excerpt(doc) -> str:
    body = text_for(doc)
    match = re.search(r"\bItem\s+\d\.\d{2}\b", body, flags=re.IGNORECASE)
    if match:
        body = body[match.start() :]
    return body[:3200] + ("…" if len(body) > 3200 else "")


with st.container(key="workspace_header"):
    brand, company_picker = st.columns([4.4, 1.6], vertical_alignment="bottom")
    with brand:
        html('<div class="brand">argus<span>RESEARCH</span></div>')

dataset_path = (
    Path(os.environ["ARGUS_DATASET"]) if os.getenv("ARGUS_DATASET") else ROOT / "data/pilot"
)
if not (dataset_path / "manifest.json").exists():
    st.error("Company data is not available yet.")
    st.stop()
try:
    dataset = Dataset(dataset_path)
except (ValueError, OSError) as exc:
    st.error(f"Cannot load dataset: {exc}")
    st.stop()

with (
    company_picker,
    st.popover(
        "Company Selection",
        width="stretch",
        help="Choose a company to explore its prices and disclosures.",
    ),
):
    ticker = st.radio(
        "Company selection",
        dataset.tickers,
        label_visibility="collapsed",
        key=f"company-{dataset.manifest.dataset_id}",
    )

company_docs = [d for d in dataset.documents if d.ticker == ticker]
company_name = company_docs[0].company.title()
if ticker == "AMD":
    company_name = "Advanced Micro Devices"
elif ticker == "CRWD":
    company_name = "CrowdStrike"
try:
    analysis = load_supplied_analysis(dataset, ticker)
except (OSError, ValueError, KeyError):
    st.error("The supplied analysis could not be loaded.")
    st.stop()
records = analysis.records if analysis else {}
preview = analysis.preview if analysis else False
status = "Fictional demo" if dataset.manifest.kind == "fictional" else ""
html(
    f'<div class="page-heading"><div><div class="eyebrow">DISCLOSURE RESEARCH</div>'
    f'<h1>{safe(company_name)} <span class="ticker">{safe(ticker)}</span></h1>'
    "<p>Follow the price. Find the disclosure. Understand what’s new.</p></div></div>"
)

prices = [p for p in dataset.prices if p.ticker == ticker]
selected_filings = dataset.replay_filings(ticker)
selection_key = f"filing-{dataset.fingerprint}-{ticker}"
chart_filter_key = f"chart-filter-{dataset.fingerprint}-{ticker}"
chart_event_key = f"chart-event-{dataset.fingerprint}-{ticker}"
with st.container(border=True, key="price_panel"):
    price_heading, clear_controls = st.columns([4, 1], vertical_alignment="center")
    with price_heading:
        html('<div class="eyebrow">PRICE HISTORY <span class="muted">/ DAILY CLOSE</span></div>')
        st.caption("Click a filing marker or drag to select a period.")
    with clear_controls:
        if st.button(
            "Clear selection", disabled=chart_filter_key not in st.session_state, width="stretch"
        ):
            st.session_state.pop(chart_filter_key, None)
            st.session_state.pop(selection_key, None)
            st.rerun()
    if prices:
        days = [p.date for p in prices]
        replay_days = [
            d for d in days if dataset.manifest.replay_start <= d <= dataset.manifest.replay_end
        ] or days
        default_period = (replay_days[0], replay_days[-1])
        start, end, focused_filing = st.session_state.get(chart_filter_key, (*default_period, None))
        chart = go.Figure(
            go.Scatter(
                x=[str(p.date) for p in prices],
                y=[p.close for p in prices],
                mode="lines",
                line={"color": "#326554", "width": 2},
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
            )
        )
        events = []
        for filing in dataset.replay_filings(ticker):
            session = dataset.effective_session(filing)
            price = next((p for p in prices if p.date == session), None)
            if price:
                events.append((price, filing))
        if events:
            chart.add_trace(
                go.Scatter(
                    x=[str(p.date) for p, _ in events],
                    y=[p.close for p, _ in events],
                    mode="markers",
                    marker={
                        "size": 11,
                        "color": [
                            "#bd7939" if f.filing_id == focused_filing else "#326554"
                            for _, f in events
                        ],
                        "line": {"width": 2, "color": "white"},
                    },
                    customdata=[f.filing_id for _, f in events],
                    text=[filing_title(f) for _, f in events],
                    hovertemplate="%{x|%b %d}<br>%{text}<extra>8-K</extra>",
                )
            )
        chart.update_layout(
            height=220,
            margin={"l": 6, "r": 48, "t": 12, "b": 30},
            dragmode="select",
            clickmode="event",
            selectdirection="h",
            selections=[],
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Arial, sans-serif", "color": "#858b85", "size": 11},
            xaxis={"showgrid": False, "zeroline": False, "nticks": 6},
            yaxis={
                "side": "right",
                "gridcolor": "#eef0eb",
                "zeroline": False,
                "tickprefix": "$",
                "nticks": 4,
            },
        )
        if (start, end) != default_period:
            chart.add_vrect(
                x0=str(start),
                x1=str(end),
                fillcolor="#326554",
                opacity=0.07,
                line_width=0,
                layer="below",
            )
        event = chart_component()(
            figure=chart.to_json(),
            key=f"prices-{dataset.fingerprint}-{ticker}",
            default=None,
        )
        if event and event.get("event_id") != st.session_state.get(chart_event_key):
            st.session_state[chart_event_key] = event.get("event_id")
            resolved = resolve_selection(
                event, {f.filing_id: p.date for p, f in events}, (days[0], days[-1])
            )
            if resolved:
                st.session_state[chart_filter_key] = resolved
                if resolved[2]:
                    st.session_state[selection_key] = resolved[2]
                else:
                    st.session_state.pop(selection_key, None)
                st.rerun()
        selected_prices = [p for p in prices if start <= p.date <= end]
        first_index = days.index(selected_prices[0].date) if selected_prices else 0
        move = (
            100 * (selected_prices[-1].close / prices[first_index - 1].close - 1)
            if selected_prices and first_index > 0
            else None
        )
        close_label = (
            f"${selected_prices[-1].close:,.2f}" if selected_prices else "No trading session"
        )
        change = f"{move:+.2f}%" if move is not None else "Return unavailable"
        direction = "positive" if move is not None and move >= 0 else "negative"
        html(
            f'<div class="chart-footer"><span><b>{close_label}</b> '
            f'<span class="{direction}">{change}</span> <span class="muted">over selected dates</span>'
            f"</span><span>{start:%b %d} to {end:%b %d, %Y} "
            '<span class="legend-dot"></span> 8-K released</span></div>'
        )
        selected_filings = dataset.filings_for_window(ticker, start, end)
    else:
        st.info("No price data in this corpus. Browse its disclosures below.")

html(
    f'<div class="section-heading"><h2>Disclosures <span>{len(selected_filings):02d}</span>'
    f'</h2><span class="run-status">{safe(status)}</span></div>'
)
st.caption("Matched by release time, not proven cause. After-hours filings appear next session.")
selected = None
if not selected_filings:
    st.info("No matching 8-K in the supplied history for this window.")
else:
    selected_filings = sorted(selected_filings, key=lambda f: f.available_at, reverse=True)
    if st.session_state.get(selection_key) not in [f.filing_id for f in selected_filings]:
        st.session_state[selection_key] = selected_filings[0].filing_id
    feed, detail = st.columns([1, 1.75], gap="large")
    with feed, st.container(key="filing_feed"):
        for filing in selected_filings:
            active = st.session_state[selection_key] == filing.filing_id
            if st.button(
                f"{filing.available_at.astimezone(ET):%b %d} · {filing.form}  \n"
                f"{filing_title(filing)}",
                key=f"select-{filing.id}",
                width="stretch",
                type="primary" if active else "secondary",
            ):
                st.session_state[selection_key] = filing.filing_id
                st.rerun()
    selected = next(f for f in selected_filings if f.filing_id == st.session_state[selection_key])
    record = records.get(selected.filing_id)
    overview = ((record or {}).get("prediction") or {}).get("overview") or (record or {}).get(
        "overview"
    )
    with detail, st.container(border=True, key="filing_detail"):
        html(
            f'<div class="eyebrow">{safe(selected.form)} <span class="muted">/ '
            f"{selected.available_at.astimezone(ET):%b %d, %Y · %I:%M %p ET}</span></div>"
            f'<h2 class="detail-title">{safe(filing_title(selected))}</h2>'
        )
        source_tab, change_tab = st.tabs(["Summary", "What’s net new"])
        with source_tab:
            if overview:
                st.markdown(overview.replace("$", r"\$"))
                if note := record.get("prediction", {}).get("limitations"):
                    st.caption(note)
            else:
                html(f'<div class="source-excerpt">{safe(source_excerpt(selected))}</div>')
            if selected.source_url:
                st.link_button(
                    "Read original filing", selected.source_url, icon=":material/open_in_new:"
                )
            if overview:
                with st.expander("Filing excerpt"):
                    html(f'<div class="source-excerpt">{safe(source_excerpt(selected))}</div>')
        with change_tab:
            if preview:
                st.info("A comparison with earlier disclosures is not available for this filing.")
            elif record and record.get("prediction"):
                prediction = record["prediction"]
                if not prediction["findings"]:
                    st.info("The pipeline returned no substantive findings.")
                for index, finding in enumerate(prediction["findings"]):
                    if index:
                        st.divider()
                    labels = {
                        "new": "NEW INFORMATION",
                        "changed": "CHANGED",
                        "repeated": "ALREADY ANNOUNCED",
                        "uncertain": "UNRESOLVED",
                    }
                    st.caption(labels[finding["classification"]])
                    st.markdown("**" + finding["title"].replace("$", r"\$") + "**")
                    st.markdown(finding["announced"].replace("$", r"\$"))
                    st.markdown(finding["change"].replace("$", r"\$"))
                    with st.expander("Supporting evidence"):
                        for label, field in [
                            ("Current disclosure", "current_evidence"),
                            ("Earlier disclosure", "prior_evidence"),
                        ]:
                            if not finding[field]:
                                continue
                            st.write(f"**{label}**")
                            for evidence in finding[field]:
                                doc = next(
                                    (
                                        d
                                        for d in dataset.documents
                                        if d.id == evidence["document_id"]
                                    ),
                                    None,
                                )
                                if doc is None:
                                    st.caption("Source could not be located.")
                                    st.text(evidence["quote"])
                                    continue
                                source_label = (
                                    f"{doc.ticker} · {doc.form}"
                                    f" · {doc.available_at.astimezone(ET):%b %d, %Y}"
                                    + (" · Exhibit" if doc.role == "exhibit" else "")
                                )
                                if doc.source_url:
                                    st.markdown(f"[{source_label}]({doc.source_url})")
                                else:
                                    st.caption(source_label)
                                st.text(evidence["quote"])
                if prediction["limitations"]:
                    st.caption(prediction["limitations"])
            elif record and record["status"] == "error":
                st.error(f"Pipeline failed: {record.get('error_type', 'Unknown error')}")
            else:
                st.info("No analysis is available for this filing.")
