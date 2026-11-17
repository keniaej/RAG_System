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

## Technische Details

### Muss man die Dokumente in einzelne Dateien trennen?

Nein. Die PDF bleibt eine Datei. Die Trennung passiert **logisch beim Einlesen**:
Jede Seite hat eine Fußzeile wie `VB-2019-003 · Canis & Felis Foods B.V. · vertraulich`.
Darüber wird jede Seite ihrem Dokument zugeordnet (`split_pdf_into_documents`), die Fußzeile
selbst wird entfernt.

Wichtig ist diese Zuordnung trotzdem – nicht wegen der Dateien, sondern wegen der
**Quellenangabe und der Trennschärfe**:

- Ohne Trennung könnte ein Chunk das Ende von Dokument A und den Anfang von Dokument B
  enthalten. Werte zweier Kunden würden vermischt.
- Als Quelle käme nur „Alle_Demo-Dokumente.pdf, Seite 26“ heraus. Mit Trennung wird
  `VB-2019-003, Fleischemulsion für Trockenfutter-Coating (Canis & Felis, S. 25–26)` zitiert.

Liegen die Dokumente als Einzeldateien vor, funktioniert die Ingestion genauso
(`python -m rag ingest data/raw/`) – die Dokument-Nr. kommt immer aus der Fußzeile.

### Wie wird gechunkt?

**Nicht pro Absatz und nicht nach fester Zeichenzahl**, sondern entlang der Dokumentstruktur:

| Chunk-Typ     | Inhalt                                                           | Anzahl | Größe (Wörter, Median) |
|---------------|------------------------------------------------------------------|-------:|-----------------------:|
| `maschine`    | Kopfdaten + Maschinenkonfiguration + Liste der Produkte          | 30     | ~120                   |
| `produkt`     | ein komplettes Verfahren: Rohstoffe, Temperaturen, Drehzahl, Durchsatz, Schneidsatz-Tabelle, Prozessparameter-Tabelle, Besonderheit | 80 | ~130 (max. 175) |
| `bemerkungen` | Bemerkungen bzw. Änderungshistorie                               | 30     | ~40                    |

Erkannt werden die Grenzen über Überschriften-Muster (`ingest.py`):
`1. Produktname` / `2.1 Produktname` (Produktbeginn), `Maschinenausstattung` /
`1 Maschinenkonfiguration`, `3 Bemerkungen` / `3 Änderungshistorie`.

Eine Frage wie „Welche Lochplatte in Stufe 2 bei Kunde X für Produkt Y?“ braucht genau
*einen* Produkt-Chunk – alle Werte stehen zusammen. Absatz-Chunking würde Fließtext und
Schneidsatz-Tabelle trennen; dann fehlt einem Teil die Info, zu welchem Produkt er gehört.

Jeder Chunk bekommt zusätzlich einen **Kontext-Header** vorangestellt:

```
Dokument VB-2023-008 (Verfahrensbeschreibung) | Kunde: Nordic Meats AB, Jönköping (SE) |
Maschine: KS Feinstzerkleinerer FD225 D, Seriennummer 23978, Baujahr 2022 | Datum: 23.10.2023
Produkt/Verfahren: Lyoner-Brät
2.2 Lyoner-Brät
Rohstoffe: ...
```

Die Chunks sind mit ~100–175 Wörtern klein genug für präzise Suche und groß genug, um ein
Verfahren vollständig zu enthalten. Alle Chunks ansehen: `python -m rag export chunks.jsonl`.

### Wie werden die Tabellen behandelt?

Tabellen werden **nicht separat gechunkt**, sondern bleiben als Textzeilen in dem Chunk, zu
dem sie gehören (Maschinenausstattung → `maschine`, Schneidsatz und Prozessparameter →
`produkt`). `pdfplumber` gibt jede Tabellenzeile als eine Textzeile aus, inklusive Kopfzeile:

```
Schneidsatzbestückung
Stufe Typ Bohrung / Schneidspalt Bemerkung
1 Messerkopf / Lochplatte Lochplatte 8,0 mm Messerkopf 6-Klingen
2 Messerkopf / Lochplatte Lochplatte 3,0 mm
3 RotaPlate Bohrung 0,6 mm
```

Weil die Kopfzeile mitkommt, kann das LLM die Spalten zuordnen. Bei 49 von 80 Produkten
läuft eine Tabelle über einen Seitenumbruch (z. B. „Prozessparameter“ unten auf Seite 25,
„Parameter / Wert“ oben auf Seite 26). Da die Seiten eines Dokuments vor dem Chunking
zusammengefügt und die Fußzeilen entfernt werden, landet die Tabelle trotzdem vollständig in
einem Chunk.

Zusätzlich werden einige Kennzahlen per Regex als Metadaten extrahiert
(`durchsatz_kg_h`, `drehzahl_min`, `vakuum`, `stufen`; t/h wird in kg/h umgerechnet).
Sie dienen zum Filtern und als Basis für spätere Tabellen-Abfragen.

### Was ist `k`?

`k` = **Anzahl der Chunks, die die Suche zurückgibt und an das LLM übergibt** (Standard 6,
CLI `-k`, im UI ein Schieberegler).

- zu klein (1–2): bei Vergleichsfragen („Welche Kunden fahren Hummus?“, 5 Treffer) fehlen Quellen
- zu groß (20+): mehr irrelevanter Kontext, höhere Kosten, das LLM wird eher abgelenkt

In der Evaluation bedeutet **Hit@k**: Anteil der Fragen, bei denen der richtige Chunk unter
den ersten k Treffern ist. Nicht verwechseln mit `rrf_k = 60` in `index.py` – das ist eine
Glättungskonstante der Reciprocal Rank Fusion (Score = Σ 1 / (60 + Rang)), die BM25- und
Vektor-Rangliste zusammenführt.

### Welches LLM / welche Modelle?

| Aufgabe     | Modell                                           | konfigurierbar über |
|-------------|--------------------------------------------------|---------------------|
| Antwort (lokal) | **Ollama**, Standard `llama3.2:3b`             | `RAG_LLM_BACKEND=ollama`, `RAG_OLLAMA_MODEL` |
| Antwort (Cloud) | **Claude** über die Anthropic API, Standard `claude-sonnet-5-5` | `RAG_LLM_BACKEND=claude`, `RAG_LLM_MODEL` |
| Embeddings  | `intfloat/multilingual-e5-small` (lokal, mehrsprachig) | `RAG_EMBED_MODEL`, `RAG_EMBEDDER` |
| Fallback    | TF-IDF (Zeichen-3- bis 5-Gramme) statt Embeddings; extraktive Antwort statt LLM | automatisch |

Das LLM wird nur für die *Formulierung* der Antwort genutzt. Suche und Chunking laufen lokal
ohne LLM. Ein weiteres LLM (z. B. OpenAI) lässt sich anbinden, indem man in
`generator.py` eine Klasse mit derselben `generate(question, hits)`-Methode ergänzt.

Die Evaluationswerte unten wurden mit dem TF-IDF-Fallback gemessen; mit dem Embedding-Modell
sind bessere Werte bei umschriebenen Fragen zu erwarten.

### Ablauf einer Frage

1. Frage auf Dokument-Nr. oder eindeutigen Kundennamen prüfen → ggf. Filter setzen
2. BM25-Ranking und Vektor-Ranking über alle (gefilterten) Chunks
3. Fusion per RRF → Top-k Chunks
4. Prompt an Claude: Systemregeln (nur aus Quellen, Werte exakt, Quelle zitieren) + k Chunks + Frage
5. Antwort + Quellenliste (Dokument, Produkt, Kunde, Seiten)

## Schnellstart (lokal mit Ollama, ohne API-Key)

Voraussetzung: [Ollama](https://ollama.com) ist installiert und läuft (Ollama-App geöffnet
oder `ollama serve` in einem eigenen Terminal).

```bash
ollama list                     # Modellnamen prüfen, z. B. llama3.2:3b
ollama pull llama3.2:3b         # nur falls noch nicht vorhanden

cd rag_system
conda create -n rag python=3.11 -y && conda activate rag     # oder: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # RAG_OLLAMA_MODEL = Name aus `ollama list`

export PYTHONPATH=src           # oder einmalig: pip install -e .
python -m rag ingest            # PDF einlesen, Index bauen (lädt beim 1. Mal das Embedding-Modell)
python -m rag ask "Welche Maschine produziert Nassfutter-Paté?"
python -m rag chat              # interaktiv Fragen stellen
streamlit run app.py            # Web-Oberfläche im Browser
```

Mit Claude statt Ollama: in `.env` `RAG_LLM_BACKEND=claude` und `ANTHROPIC_API_KEY` setzen.

**Hinweis zu kleinen Modellen (3B):** Die Suche ist vom LLM unabhängig, die Quellen stimmen also
immer. Ein 3B-Modell formuliert aber schwächer, vor allem bei Vergleichen über viele Kunden,
und kann Zahlen vertauschen. Darum immer die angezeigten Quellen gegenprüfen. Bessere lokale
Alternativen, falls der Rechner es schafft: `llama3.1:8b`, `qwen2.5:7b`, `mistral-nemo`.

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
