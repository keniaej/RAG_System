"""Hybrid-Index: BM25 (lexikalisch) + Vektorsuche (semantisch), fusioniert per RRF.

Gerade bei technischen Dokumenten sind exakte Treffer wichtig (Dokument-Nr.,
Seriennummern, "FL248 SE", "1.4404", "Lochplatte 1,5 mm") – dafür BM25. Für
umschriebene Fragen ("Welche Kunden machen Käse?") hilft die Vektorsuche.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .embeddings import Embedder, load_embedder, make_embedder
from .ingest import Chunk, load_chunks, save_chunks

TOKEN_RE = re.compile(r"vb-\d{4}-\d{3}|\d+(?:[.,]\d+)*|[a-zäöüß]+(?:\d+)?|\w+", re.UNICODE)
STOPWORDS = set(
    """der die das den dem des ein eine einer eines einem einen und oder mit bei im in am an auf
    aus für von vom zu zur zum ist sind wird werden wurde wurden welche welcher welches wie was wer
    wo es sie er nicht auch als über per laut lt nach dabei beim""".split()
)

# Namensbestandteile, die auch als normale Wörter in Fragen vorkommen und daher
# keinen Kundenfilter auslösen sollen.
GENERIC_NAME_TOKENS = set(
    """gmbh foods food petfood pet metzgerei hofmetzgerei käserei dairy processing feinkost
    delicatessen wurstmanufaktur babyfood charcuterie fromagerie premium sauces salsas nutrition
    craft choice harvest surimi veggie nordic""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        out.append(t)
        # Bindestrich-Komposita zusätzlich zerlegen: "formfleisch-emulsion"
        if "-" in t and not t.startswith("vb-"):
            out.extend(p for p in t.split("-") if p)
    return out


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.doc_tf = [Counter(d) for d in docs]
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.doc_tf), dtype=np.float32)
        for i, tf in enumerate(self.doc_tf):
            denom_base = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            for t in query:
                f = tf.get(t)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + denom_base)
        return s


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25().fit([tokenize(c.text) for c in chunks])
        if vectors is None:
            embedder.fit([c.text for c in chunks])
            vectors = embedder.embed_documents([c.text for c in chunks])
        self.vectors = vectors

    # -------------------------------------------------------------- build/IO
    @classmethod
    def build(cls, chunks: list[Chunk], embedder_kind: str = "auto", model_name: str | None = None) -> "HybridIndex":
        return cls(chunks, make_embedder(embedder_kind, model_name))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_chunks(self.chunks, d / "chunks.jsonl")
        np.save(d / "vectors.npy", self.vectors)
        (d / "embedder.bin").write_bytes(self.embedder.dumps())
        (d / "meta.json").write_text(
            json.dumps({"embedder": self.embedder.name, "n_chunks": len(self.chunks)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "HybridIndex":
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"Kein Index in {d} – zuerst `python -m rag ingest` ausführen.")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        embedder = load_embedder(meta["embedder"], (d / "embedder.bin").read_bytes())
        return cls(load_chunks(d / "chunks.jsonl"), embedder, np.load(d / "vectors.npy"))

    # ---------------------------------------------------------------- search
    def search(
        self,
        query: str,
        k: int = 6,
        filters: dict | None = None,
        rrf_k: int = 60,
        candidates: int = 40,
    ) -> list[tuple[Chunk, float]]:
        mask = self._filter_mask(filters or {})
        if not mask.any():
            return []

        bm = self.bm25.scores(tokenize(query))
        vec = self.vectors @ self.embedder.embed_query(query)

        fused = np.zeros(len(self.chunks), dtype=np.float32)
        for scores in (bm, vec):
            s = np.where(mask, scores, -np.inf)
            order = [i for i in np.argsort(-s)[:candidates] if np.isfinite(s[i])]
            for rank, i in enumerate(order):
                fused[i] += 1.0 / (rrf_k + rank + 1)
        fused = np.where(mask, fused, -1)
        top = [i for i in np.argsort(-fused)[:k] if fused[i] > 0]
        return [(self.chunks[i], float(fused[i])) for i in top]

    def _filter_mask(self, filters: dict) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        for key, wanted in filters.items():
            if wanted in (None, "", []):
                continue
            wanted_list = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            wanted_norm = [str(w).lower() for w in wanted_list]
            for i, c in enumerate(self.chunks):
                val = c.doc_id if key == "doc_id" else c.metadata.get(key)
                val = str(val).lower() if val is not None else ""
                if not any(w == val or w in val for w in wanted_norm):
                    mask[i] = False
        return mask

    # ------------------------------------------------------- query analysis
    def detect_filters(self, query: str) -> dict:
        """Erkennt Dokument-Nummern und eindeutige Kundennamen in der Frage.

        "Was fährt Nordic Meats auf Stufe 3?" -> {"doc_id": ["VB-2023-008"]}
        """
        q = query.lower()
        doc_ids = sorted(set(m.upper() for m in re.findall(r"vb-\d{4}-\d{3}", q)))
        if doc_ids:
            return {"doc_id": doc_ids}

        customers: dict[str, str] = {}
        for c in self.chunks:
            customers.setdefault(c.doc_id, c.metadata.get("kunde_name", ""))
        token_owner: dict[str, set[str]] = {}
        for doc_id, name in customers.items():
            for t in tokenize(name):
                if len(t) >= 4:
                    token_owner.setdefault(t, set()).add(doc_id)
        q_tokens = set(tokenize(query)) - GENERIC_NAME_TOKENS
        hits = set()
        for t in q_tokens:
            owners = token_owner.get(t)
            # nur Tokens, die genau einen Kunden identifizieren (nicht "petfood", "gmbh" ...)
            if owners and len(owners) == 1:
                hits |= owners
        return {"doc_id": sorted(hits)} if hits else {}


def chunk_to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)
