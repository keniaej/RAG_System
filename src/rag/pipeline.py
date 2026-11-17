"""High-Level-API: Index bauen, suchen, fragen."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .generator import make_generator
from .index import HybridIndex
from .ingest import Chunk, ingest_paths


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimaler .env-Loader (ohne Zusatzpaket); vorhandene Variablen bleiben unangetastet."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()
DEFAULT_DATA = Path(os.getenv("RAG_DATA_DIR", "data/raw"))
DEFAULT_INDEX = Path(os.getenv("RAG_INDEX_DIR", "index"))


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[tuple[Chunk, float]]
    filters: dict


def build_index(
    paths: list[str | Path] | None = None,
    index_dir: str | Path = DEFAULT_INDEX,
    embedder: str = os.getenv("RAG_EMBEDDER", "auto"),
    model_name: str | None = os.getenv("RAG_EMBED_MODEL"),
) -> HybridIndex:
    chunks = ingest_paths(paths or [DEFAULT_DATA])
    index = HybridIndex.build(chunks, embedder, model_name)
    index.save(index_dir)
    return index


class RAG:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX, generator: str = "auto") -> None:
        self.index = HybridIndex.load(index_dir)
        self.generator = make_generator(generator)

    def retrieve(self, question: str, k: int = 6, filters: dict | None = None, auto_filter: bool = True):
        used = dict(filters or {})
        if auto_filter and not used:
            used = self.index.detect_filters(question)
        hits = self.index.search(question, k=k, filters=used)
        if not hits and used and not filters:  # automatischer Filter war zu streng
            used = {}
            hits = self.index.search(question, k=k)
        return hits, used

    def ask(self, question: str, k: int = 6, filters: dict | None = None, auto_filter: bool = True) -> Answer:
        hits, used = self.retrieve(question, k=k, filters=filters, auto_filter=auto_filter)
        return Answer(question, self.generator.generate(question, hits), hits, used)
