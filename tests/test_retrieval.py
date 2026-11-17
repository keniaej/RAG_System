from pathlib import Path

import pytest

from rag.generator import ExtractiveGenerator
from rag.index import HybridIndex, tokenize
from rag.ingest import ingest_pdf

PDF = Path(__file__).resolve().parents[1] / "data" / "raw" / "Alle_Demo-Dokumente.pdf"


@pytest.fixture(scope="module")
def index():
    return HybridIndex.build(ingest_pdf(PDF), embedder_kind="tfidf")


def test_tokenize_keeps_doc_ids_and_numbers():
    toks = tokenize("Was steht in VB-2019-003 zur Lochplatte 1,5 mm und Formfleisch-Emulsion?")
    assert "vb-2019-003" in toks
    assert "1,5" in toks
    assert "formfleisch" in toks and "emulsion" in toks


def test_detect_customer_filter(index):
    assert index.detect_filters("Welche Motorleistung hat die Maschine bei Nordic Meats?") == {
        "doc_id": ["VB-2023-008"]
    }
    assert index.detect_filters("Infos zu VB-2019-003") == {"doc_id": ["VB-2019-003"]}
    # generische Wörter lösen keinen Filter aus
    assert index.detect_filters("Welche Metzgerei fährt Lyoner mit Vakuum?") == {}


def test_search_finds_serial_number(index):
    hits = index.search("Seriennummer 28565", k=3)
    assert hits[0][0].doc_id == "VB-2019-003"


def test_search_with_filter(index):
    hits = index.search("Lyoner Durchsatz", k=5, filters={"doc_id": ["VB-2023-008"]})
    assert hits and all(c.doc_id == "VB-2023-008" for c, _ in hits)
    assert "VB-2023-008#p2" in [c.chunk_id for c, _ in hits[:2]]


def test_save_and_load(index, tmp_path):
    index.save(tmp_path)
    loaded = HybridIndex.load(tmp_path)
    q = "Reklamation sichtbare Körnung"
    assert [c.chunk_id for c, _ in loaded.search(q)] == [c.chunk_id for c, _ in index.search(q)]


def test_extractive_generator_cites_sources(index):
    hits = index.search("Hummus Durchsatz", k=3)
    text = ExtractiveGenerator().generate("Hummus Durchsatz", hits)
    assert "[VB-" in text and "Hummus" in text
