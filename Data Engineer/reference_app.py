"""Read-only development references. This is not the investor's model output."""

from pathlib import Path

import streamlit as st

from net_new.dataset import Dataset
from net_new.references import load_references

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Reference examples · Argus", layout="wide")
st.html(f"<style>{(ROOT / 'ui.css').read_text(encoding='utf-8')}</style>")
st.caption("ARGUS · DEVELOPMENT REFERENCES")
st.title("Reference examples")
st.caption(
    "Expected facts and supporting sources. These are authored references, not pipeline predictions. Wording may vary; coverage is non-exhaustive."
)
try:
    dataset = Dataset(ROOT / "data/pilot")
    cases = load_references(ROOT / "references/development.jsonl", dataset)
except (OSError, ValueError) as exc:
    st.error(f"Reference dataset unavailable: {exc}")
    st.stop()
selected = st.selectbox(
    "Disclosure",
    range(len(cases)),
    format_func=lambda i: f"{cases[i].ticker} · {cases[i].available_at[:10]} · {cases[i].title}",
)
case = cases[selected]
st.subheader(case.title)
st.write(case.overview.replace("$", r"\$"))
for item in case.findings:
    with st.container(border=True):
        st.markdown("**" + item.title + "**")
        st.caption("Acceptable labels: " + ", ".join(item.acceptable_classifications))
        for fact in item.expected_facts:
            st.markdown(fact.replace("$", r"\$"))
        st.markdown(item.comparison.replace("$", r"\$"))
        with st.expander("Evidence and interpretation"):
            for label, evidence in [
                ("Current disclosure", item.current_evidence),
                ("Earlier disclosure", item.prior_evidence),
            ]:
                for citation in evidence:
                    doc = next(d for d in dataset.documents if d.id == citation.document_id)
                    st.markdown(
                        f"[{label} · {doc.form} · {doc.available_at:%Y-%m-%d}]({doc.source_url})"
                    )
                    st.text(citation.quote)
            st.write(item.adjudication_notes)
            st.write("Acceptable variations: " + " ".join(item.acceptable_variations))
with st.expander("Coverage and review notes"):
    st.write(case.other_supported_topics)
    for note in case.limitations:
        st.write(note)
    st.caption(case.case_id + " · Reference " + case.reference_version)
