from pathlib import Path

import pytest

from rag.ingest import ingest_pdf, normalize, parse_header

PDF = Path(__file__).resolve().parents[1] / "data" / "raw" / "Alle_Demo-Dokumente.pdf"


@pytest.fixture(scope="module")
def chunks():
    return ingest_pdf(PDF)


def test_all_documents_found(chunks):
    doc_ids = {c.doc_id for c in chunks}
    assert len(doc_ids) == 30
    assert "VB-2015-012" in doc_ids and "VB-2025-025" in doc_ids


def test_every_doc_has_machine_chunk_and_products(chunks):
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c.section)
    for doc_id, sections in by_doc.items():
        assert sections.count("maschine") == 1, doc_id
        assert sections.count("produkt") >= 2, doc_id


def test_header_metadata(chunks):
    m = next(c for c in chunks if c.chunk_id == "VB-2023-008#maschine").metadata
    assert m["kunde_name"] == "Nordic Meats AB"
    assert m["land"] == "SE"
    assert m["maschinentyp"] == "FD225 D"
    assert m["seriennummer"] == "23978"
    assert m["baujahr"] == 2022
    assert m["seiten"] == [54, 55]


def test_product_chunk_content_and_facts(chunks):
    c = next(c for c in chunks if c.chunk_id == "VB-2023-008#p2")
    assert c.title == "Lyoner-Brät"
    assert "RotaPlate Bohrung 0,6 mm" in c.text
    assert c.metadata["durchsatz_kg_h"] == 1600
    assert c.metadata["vakuum"] is True
    assert c.text.startswith("Dokument VB-2023-008")  # Kontext-Header


def test_throughput_in_tons_is_converted(chunks):
    c = next(c for c in chunks if c.chunk_id == "VB-2023-029#p1")
    assert c.metadata["durchsatz_kg_h"] == 7000


def test_old_layout_is_normalized(chunks):
    c = next(c for c in chunks if c.chunk_id == "VB-2015-012#p1")
    assert "Lochplatte 1,5 mm" in c.text
    assert "-0,3 °C" in c.text
    assert "LP 1,5 mm" in c.raw_text  # Original bleibt erhalten


def test_normalize():
    assert normalize("2400 U/min, 7,6 Grad C, MK / LP LP 2,0 mm MK 3-Klingen") == (
        "2400 min⁻¹, 7,6 °C, Messerkopf / Lochplatte Lochplatte 2,0 mm Messerkopf 3-Klingen"
    )


def test_parse_header_both_layouts():
    old = parse_header(["Kunde: PetVital Nutrition GmbH, Cloppenburg (DE)", "Baujahr: 2015"])
    new = parse_header(["Kunde Nordic Meats AB, Jönköping (SE)", "Baujahr 2022"])
    assert old["kunde_name"] == "PetVital Nutrition GmbH" and old["baujahr"] == 2015
    assert new["ort"] == "Jönköping" and new["baujahr"] == 2022
