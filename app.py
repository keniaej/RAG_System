"""Streamlit-Oberfläche:  streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag.pipeline import DEFAULT_INDEX, RAG, build_index  # noqa: E402

st.set_page_config(page_title="RAG – Verfahrensbeschreibungen", layout="wide")


@st.cache_resource
def load_rag() -> RAG:
    if not (Path(DEFAULT_INDEX) / "meta.json").exists():
        build_index()
    return RAG(DEFAULT_INDEX)


rag = load_rag()
chunks = rag.index.chunks
kunden = sorted({c.metadata.get("kunde_name", "") for c in chunks})
typen = sorted({c.metadata.get("maschinentyp", "") for c in chunks})

with st.sidebar:
    st.header("Filter")
    f_kunde = st.selectbox("Kunde", ["(alle)"] + kunden)
    f_typ = st.selectbox("Maschinentyp", ["(alle)"] + typen)
    k = st.slider("Anzahl Quellen (k)", 2, 15, 6)
    auto = st.checkbox("Kunde/Dokument automatisch aus Frage erkennen", value=True)
    st.caption(f"{len({c.doc_id for c in chunks})} Dokumente · {len(chunks)} Chunks · Embedder: {rag.index.embedder.name}")
    st.caption(f"Generator: {type(rag.generator).__name__}")

st.title("Fragen an die Verfahrensbeschreibungen")
question = st.text_input("Frage", placeholder="z. B. Welche Schneidsatzbestückung fährt Nordic Meats für Lyoner-Brät?")

if question:
    filters = {}
    if f_kunde != "(alle)":
        filters["kunde_name"] = f_kunde
    if f_typ != "(alle)":
        filters["maschinentyp"] = f_typ
    with st.spinner("Suche und formuliere Antwort …"):
        ans = rag.ask(question, k=k, filters=filters or None, auto_filter=auto)
    if ans.filters:
        st.caption(f"Aktive Filter: {ans.filters}")
    st.markdown(ans.answer)
    st.subheader("Quellen")
    for c, score in ans.sources:
        with st.expander(f"{c.doc_id} · {c.title} · {c.metadata.get('kunde_name')} · S. {c.metadata.get('seiten')}  ({score:.3f})"):
            st.text(c.text)
