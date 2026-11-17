"""Ingestion: Sammel-PDF -> einzelne Verfahrensbeschreibungen -> strukturierte Chunks.

Die Demo-PDF enthält viele Verfahrensbeschreibungen (VB-JJJJ-NNN) hintereinander,
in drei unterschiedlichen Layout-Generationen. Jede Seite trägt eine Fußzeile
``VB-2019-003 · Kunde · vertraulich``; darüber werden die Seiten den Dokumenten
zugeordnet. Anschließend wird jedes Dokument entlang seiner Struktur zerlegt:

* ein Chunk für Kopfdaten + Maschinenkonfiguration,
* ein Chunk pro gefahrenem Produkt / Verfahren,
* ein Chunk für Bemerkungen / Änderungshistorie (falls vorhanden).

Jeder Chunk bekommt einen Kontext-Header (Dokument, Kunde, Maschine, Produkt),
damit er auch isoliert eindeutig zuordenbar ist.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

FOOTER_RE = re.compile(r"^(VB-\d{4}-\d{3})\s*·\s*(.+?)\s*·\s*(intern|vertraulich)\s*$")
HEADER_FIELD_RE = re.compile(
    r"^(Kunde|Maschine|Seriennummer|Baujahr|Erstellt von|Datum|Dokument-Nr\.|Revision)\s*:?\s+(.+)$"
)
# "1. Nassfutter-Paté Rind" (alte Layouts) oder "2.1 Formfleisch-Emulsion" (neuere Layouts)
PRODUCT_RE = re.compile(r"^(?:(\d{1,2})\.|2\.(\d{1,2}))\s+([A-Za-zÄÖÜäöü].+)$")
MACHINE_START_RE = re.compile(r"^(Maschinenausstattung|1 Maschinenkonfiguration)\s*$")
TRAILER_RE = re.compile(
    r"^(3 (Bemerkungen|Änderungshistorie)\s*$|Stand nach Inbetriebnahme|Werte lt\. Aufnahme)"
)
SKIP_LINE_RE = re.compile(r"^(Gefahrene Produkte|2 Verfahren)\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str  # "maschine" | "produkt" | "bemerkungen"
    title: str
    text: str  # normalisierter Text inkl. Kontext-Header (für Index + LLM)
    raw_text: str  # Originaltext der Sektion
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalisierung: ältere Dokumente nutzen Abkürzungen ("MK", "LP", "Grad C",
# "U/min"). Für ein einheitliches Retrieval werden sie vereinheitlicht.
# ---------------------------------------------------------------------------
_NORMALIZE = [
    (re.compile(r"(\d)\n(°C|Grad C|%|kg/h|mm)"), r"\1 \2"),  # PDF-Zeilenumbruch vor Einheit
    (re.compile(r"\bMK / LP\b"), "Messerkopf / Lochplatte"),
    (re.compile(r"\bLP (?=\d)"), "Lochplatte "),
    (re.compile(r"\bMK (?=\d)"), "Messerkopf "),
    (re.compile(r"(\d)\s*Grad C\b"), r"\1 °C"),
    (re.compile(r"\bGrad C\b"), "°C"),
    (re.compile(r"\bU/min\b"), "min⁻¹"),
]


def normalize(text: str) -> str:
    for pattern, repl in _NORMALIZE:
        text = pattern.sub(repl, text)
    return text


def _parse_number(value: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d{3})*(?:,\d+)?", value)
    if not m:
        return None
    return float(m.group(0).replace(".", "").replace(",", "."))


# ---------------------------------------------------------------------------
# PDF -> Dokumente
# ---------------------------------------------------------------------------
def split_pdf_into_documents(pdf_path: str | Path) -> list[dict]:
    """Gruppiert Seiten anhand der Fußzeile zu Dokumenten."""
    docs: dict[str, dict] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            lines = [l.rstrip() for l in (page.extract_text() or "").splitlines()]
            footer = None
            body = []
            for line in lines:
                m = FOOTER_RE.match(line.strip())
                if m:
                    footer = m
                else:
                    body.append(line)
            if footer is None:
                raise ValueError(f"Seite {page_no}: keine Dokument-Fußzeile gefunden")
            doc_id, customer_short, classification = footer.groups()
            doc = docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "customer_short": customer_short,
                    "classification": classification,
                    "pages": [],
                    "lines": [],
                },
            )
            doc["pages"].append(page_no)
            doc["lines"].extend(body)
    return list(docs.values())


def parse_header(lines: list[str]) -> dict:
    meta: dict = {}
    for line in lines:
        m = HEADER_FIELD_RE.match(line.strip())
        if m and m.group(1) not in meta:
            meta[m.group(1)] = m.group(2).strip()
    out = {
        "kunde": meta.get("Kunde", ""),
        "maschine": meta.get("Maschine", ""),
        "seriennummer": meta.get("Seriennummer", ""),
        "baujahr": int(meta["Baujahr"]) if meta.get("Baujahr", "").isdigit() else None,
        "erstellt_von": meta.get("Erstellt von", ""),
        "datum": meta.get("Datum", ""),
        "revision": meta.get("Revision", ""),
    }
    # "Nordic Meats AB, Jönköping (SE)" -> Ort + Land
    m = re.match(r"^(.*?),\s*(.+?)\s*\((\w{2})\)$", out["kunde"])
    if m:
        out["kunde_name"], out["ort"], out["land"] = m.groups()
    else:
        out["kunde_name"], out["ort"], out["land"] = out["kunde"], "", ""
    m = re.search(r"\b(F[A-Z]\d{3}\s*\w*)$", out["maschine"])
    out["maschinentyp"] = m.group(1).strip() if m else out["maschine"]
    out["jahr"] = int(out["datum"][-4:]) if re.search(r"\d{4}$", out["datum"]) else None
    return out


def split_sections(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]], list[str]]:
    """Teilt die Zeilen eines Dokuments in Kopf/Maschine, Produkte und Trailer."""
    head: list[str] = []
    products: list[tuple[str, list[str]]] = []
    trailer: list[str] = []
    state = "head"
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if TRAILER_RE.match(line):
            state = "trailer"
            trailer.append(line)
            continue
        if state != "trailer":
            pm = PRODUCT_RE.match(line)
            if pm and (state == "products" or _looks_like_first_product(head)):
                state = "products"
                products.append((pm.group(3).strip(), [line]))
                continue
        if SKIP_LINE_RE.match(line):
            continue
        if state == "head":
            head.append(line)
        elif state == "products":
            products[-1][1].append(line)
        else:
            trailer.append(line)
    return head, products, trailer


def _looks_like_first_product(head: list[str]) -> bool:
    # Produkte beginnen erst nach dem Maschinenblock.
    return any(MACHINE_START_RE.match(l) for l in head)


def _extract_product_facts(text: str) -> dict:
    """Leichtgewichtige Extraktion einiger Kennzahlen für Filter/Anzeige."""
    facts: dict = {}
    m = re.search(r"Durchsatz (?:beträgt )?([\d.,]+) ?(kg/h|t/h)", text)
    if m:
        value = _parse_number(m.group(1))
        facts["durchsatz_kg_h"] = value * 1000 if m.group(2) == "t/h" and value is not None else value
    m = re.search(r"(\d{3,4}) ?min⁻¹", text)
    if m:
        facts["drehzahl_min"] = int(m.group(1))
    facts["vakuum"] = bool(re.search(r"Vakuum\s+ja|unter Vakuum|mit Vakuumtrichter", text))
    m = re.search(r"\b(ein|zwei|drei|vier)stufig", text)
    if m:
        facts["stufen"] = {"ein": 1, "zwei": 2, "drei": 3, "vier": 4}[m.group(1)]
    return facts


def build_chunks(doc: dict) -> list[Chunk]:
    head, products, trailer = split_sections(doc["lines"])
    meta = parse_header(head)
    base_meta = {
        "doc_id": doc["doc_id"],
        "klassifizierung": doc["classification"],
        "seiten": doc["pages"],
        **meta,
    }
    ctx = (
        f"Dokument {doc['doc_id']} (Verfahrensbeschreibung) | Kunde: {meta['kunde']} | "
        f"Maschine: {meta['maschine']}, Seriennummer {meta['seriennummer']}, Baujahr {meta['baujahr']} | "
        f"Datum: {meta['datum']}"
    )
    chunks: list[Chunk] = []

    head_raw = "\n".join(head)
    chunks.append(
        Chunk(
            chunk_id=f"{doc['doc_id']}#maschine",
            doc_id=doc["doc_id"],
            section="maschine",
            title="Kopfdaten & Maschinenkonfiguration",
            text=f"{ctx}\nAbschnitt: Kopfdaten und Maschinenkonfiguration\n"
            + "Gefahrene Produkte: " + "; ".join(p for p, _ in products) + "\n"
            + normalize(head_raw),
            raw_text=head_raw,
            metadata={**base_meta, "produkt": None, "produkte": [p for p, _ in products]},
        )
    )
    for i, (product, plines) in enumerate(products, start=1):
        raw = "\n".join(plines)
        norm = normalize(raw)
        chunks.append(
            Chunk(
                chunk_id=f"{doc['doc_id']}#p{i}",
                doc_id=doc["doc_id"],
                section="produkt",
                title=product,
                text=f"{ctx}\nProdukt/Verfahren: {product}\n{norm}",
                raw_text=raw,
                metadata={**base_meta, "produkt": product, **_extract_product_facts(norm)},
            )
        )
    if trailer:
        raw = "\n".join(trailer)
        chunks.append(
            Chunk(
                chunk_id=f"{doc['doc_id']}#bemerkungen",
                doc_id=doc["doc_id"],
                section="bemerkungen",
                title="Bemerkungen / Änderungshistorie",
                text=f"{ctx}\nAbschnitt: Bemerkungen / Änderungshistorie\n{normalize(raw)}",
                raw_text=raw,
                metadata={**base_meta, "produkt": None},
            )
        )
    return chunks


def ingest_pdf(pdf_path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in split_pdf_into_documents(pdf_path):
        chunks.extend(build_chunks(doc))
    return chunks


def ingest_paths(paths: list[str | Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.pdf")) if p.is_dir() else [p]
        for f in files:
            chunks.extend(ingest_pdf(f))
    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
