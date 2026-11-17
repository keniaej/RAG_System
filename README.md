# RAG-System für Verfahrensbeschreibungen (Feinstzerkleinerer)

Frage-Antwort-System (Retrieval-Augmented Generation) über die Demo-Verfahrensbeschreibungen
in `data/raw/Alle_Demo-Dokumente.pdf`: 30 Dokumente (VB-2015-012 … VB-2025-025), 71 Seiten,
drei unterschiedliche Layout-Generationen.

```
$ python -m rag ask "Welche Schneidsatzbestückung fährt Nordic Meats für Lyoner-Brät?"
```

## Architektur

```
PDF ──► Ingestion ──────────────► Chunks (+ Metadaten) ──► Hybrid-Index ──► Retrieval ──► Claude ──► Antwort mit Quellen
        • Seiten per Fußzeile        • 1× Kopf/Maschine        • BM25              • Auto-Filter       [VB-…, Produkt]
          (VB-JJJJ-NNN) zu             pro Dokument            • Embeddings          (Kunde / Dok.-Nr.)
          Dokumenten gruppieren      • 1× pro Produkt          • RRF-Fusion        • Metadatenfilter
        • Struktur-Chunking          • 1× Bemerkungen /
        • Normalisierung               Änderungshistorie
```

**Warum so?**

- **Struktur- statt Fenster-Chunking.** Jede Verfahrensbeschreibung besteht aus Kopfdaten,
  Maschinenkonfiguration und mehreren Produkten (Rohstoffe, Temperaturen, Durchsatz,
  Schneidsatz, Prozessparameter). Ein Chunk pro Produkt hält Schneidsatz und Parameter
  zusammen – feste Fenster würden Tabellen zerschneiden und Seitenumbrüche falsch zuordnen.
- **Kontext-Header.** Jeder Chunk beginnt mit Dokument-Nr., Kunde, Maschine, Seriennummer
  und Baujahr. So bleibt z. B. „Lyoner-Brät, 1.600 kg/h“ eindeutig einem Kunden zugeordnet.
- **Normalisierung.** Ältere Dokumente schreiben `MK / LP`, `LP 1,5 mm`, `Grad C`, `U/min`;
  neuere `Messerkopf / Lochplatte`, `°C`, `min⁻¹`. Für den Index wird vereinheitlicht,
  der Originaltext bleibt in `raw_text` erhalten.
- **Hybride Suche.** BM25 trifft exakte Begriffe (Seriennummern, `FL248 SE`, `1.4404`,
  Dokument-Nr.), die Vektorsuche umschriebene Fragen. Beide Ranglisten werden per
  Reciprocal Rank Fusion kombiniert.
- **Auto-Filter.** Nennt die Frage eine Dokument-Nr. oder einen eindeutigen Kundennamen,
  wird auf dieses Dokument eingeschränkt – verhindert, dass ähnliche Produkte anderer
  Kunden die Antwort verfälschen.
- **Grounded Generation.** Claude antwortet nur aus den Quellen, nennt Werte exakt mit
  Einheit, zitiert `[VB-…, Produkt]` und sagt ausdrücklich, wenn ein Wert „k. A.“ ist.

## Schnellstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # ANTHROPIC_API_KEY eintragen

export PYTHONPATH=src           # oder: pip install -e .
python -m rag ingest            # PDF(s) aus data/raw einlesen, Index nach index/
python -m rag ask "Welche Motorleistung hat die Maschine bei Nordic Meats?"
python -m rag chat              # interaktiv
streamlit run app.py            # Web-Oberfläche
```

Weitere PDFs einfach in `data/raw/` legen und `python -m rag ingest` erneut ausführen.

### Ohne API-Key / ohne Modell-Download

Alles Optionale fällt sauber zurück:

| Komponente  | Standard                                 | Fallback                                   |
|-------------|------------------------------------------|--------------------------------------------|
| Embeddings  | `intfloat/multilingual-e5-small`         | TF-IDF auf Zeichen-n-Grammen (`--embedder tfidf`) |
| Antwort     | Claude (`RAG_LLM_MODEL`, Standard `claude-sonnet-5-5`) | extraktiv: relevanteste Passagen mit Quelle |

## CLI

```bash
python -m rag ingest [PFAD ...] [--embedder auto|tfidf|st] [--model NAME]
python -m rag search "Frage" [-k 6] [--doc VB-2019-003] [--kunde "Nordic"] [--maschine FL248] [--produkt Hummus]
python -m rag ask    "Frage" [gleiche Filter]  [--no-auto-filter]
python -m rag export chunks.jsonl      # Chunks zur Kontrolle ansehen
```

## Projektstruktur

```
src/rag/
  ingest.py      PDF → Dokumente → Struktur-Chunks, Metadaten, Normalisierung
  embeddings.py  sentence-transformers bzw. TF-IDF-Fallback
  index.py       BM25 + Vektorindex, RRF-Fusion, Filter, Auto-Filter, Speichern/Laden
  generator.py   Prompt + Claude-Aufruf, extraktiver Fallback
  pipeline.py    High-Level-API (build_index, RAG.ask)
  __main__.py    CLI
app.py           Streamlit-UI
eval/            Testfragen mit erwarteten Quellen + Evaluationsskript
tests/           pytest
data/raw/        Quelldokumente
```

## Qualität messen

```bash
python -m pytest -q          # Unit-Tests (Parsing, Metadaten, Retrieval)
python eval/run_eval.py      # Retrieval: Hit@k und MRR auf 22 Fragen
```

Stand mit TF-IDF-Fallback (ohne semantisches Modell):
**Hit@3 = 100 %, Hit@1 = 68 %, MRR = 0,84** (22 Fragen). Die Rang-2-Treffer sind fast immer
Fälle, in denen der Maschinen-Chunk desselben Dokuments vor dem Produkt-Chunk liegt – für die
Antwort unkritisch, da beide an das LLM gehen. Neue Fragen in `eval/questions.jsonl` ergänzen.

## Grenzen & nächste Schritte

- **Aggregationsfragen** („Welche *aller* Kunden fahren mit Vakuum?“) braucht mehr als Top-k.
  Die Produkt-Chunks tragen bereits extrahierte Kennzahlen (`durchsatz_kg_h`, `drehzahl_min`,
  `vakuum`, `stufen`) – darauf lässt sich ein SQL-/Pandas-Tool für solche Fragen aufsetzen.
- Ländercodes stehen als `(FI)`, `(SE)` … im Text; Fragen nach „Finnland“ profitieren vom
  semantischen Embedding-Modell.
- Ein Cross-Encoder-Reranker (z. B. `BAAI/bge-reranker-v2-m3`) würde Hit@1 weiter verbessern.
- Neue Dokument-Layouts: Regexe in `ingest.py` (`FOOTER_RE`, `PRODUCT_RE`, `TRAILER_RE`) prüfen;
  `tests/test_ingest.py` schlägt an, wenn Dokumente oder Produkte nicht erkannt werden.

> Hinweis: Die Demo-Dokumente sind teils als „vertraulich“ gekennzeichnet. Bei echten Kundendaten
> das Repository privat halten bzw. `data/raw/` in `.gitignore` aufnehmen.
