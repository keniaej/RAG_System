"""Embedding-Backends.

* ``SentenceTransformerEmbedder`` – echtes semantisches Embedding (mehrsprachig),
  Standard: ``intfloat/multilingual-e5-small``. Benötigt ``sentence-transformers``.
* ``TfidfEmbedder`` – Fallback ohne Modell-Download: TF-IDF über Zeichen-n-Gramme.
  Robust gegenüber deutschen Komposita und Flexion ("Lochplatte"/"Lochplatten"),
  aber nicht semantisch.

Beide liefern L2-normalisierte Vektoren, Ähnlichkeit = Skalarprodukt.
"""
from __future__ import annotations

import pickle
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def dumps(self) -> bytes: ...


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True, min_df=1
        )

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def _transform(self, texts: list[str]) -> np.ndarray:
        m = self.vectorizer.transform(texts).astype(np.float32).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(norms, 1e-12)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]

    def dumps(self) -> bytes:
        return pickle.dumps(self.vectorizer)

    @classmethod
    def loads(cls, data: bytes) -> "TfidfEmbedder":
        obj = cls()
        obj.vectorizer = pickle.loads(data)
        return obj


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = f"st:{model_name}"
        self.model = SentenceTransformer(model_name)
        # E5-Modelle erwarten Präfixe
        self._e5 = "e5" in model_name.lower()

    def fit(self, texts: list[str]) -> None:  # vortrainiert, nichts zu tun
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self._e5:
            texts = [f"passage: {t}" for t in texts]
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        if self._e5:
            text = f"query: {text}"
        return np.asarray(self.model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)

    def dumps(self) -> bytes:
        return self.model_name.encode()

    @classmethod
    def loads(cls, data: bytes) -> "SentenceTransformerEmbedder":
        return cls(data.decode())


def make_embedder(kind: str = "auto", model_name: str | None = None) -> Embedder:
    """``auto``: sentence-transformers wenn installiert, sonst TF-IDF."""
    if kind in ("auto", "st", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model_name or "intfloat/multilingual-e5-small")
        except Exception as exc:  # Paket fehlt oder Modell nicht ladbar (offline)
            if kind != "auto":
                raise
            print(f"[rag] sentence-transformers nicht verfügbar ({exc.__class__.__name__}) – nutze TF-IDF-Fallback.")
    return TfidfEmbedder()


def load_embedder(name: str, data: bytes) -> Embedder:
    if name == "tfidf":
        return TfidfEmbedder.loads(data)
    return SentenceTransformerEmbedder.loads(data)
