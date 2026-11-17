"""CLI.

    python -m rag ingest [PFAD ...]            # PDFs einlesen, Index bauen
    python -m rag search "Frage" [-k 5]        # nur Retrieval anzeigen
    python -m rag ask "Frage"                  # Retrieval + Antwort
    python -m rag chat                         # interaktiver Modus
    python -m rag export chunks.jsonl          # Chunks zur Kontrolle exportieren
"""
from __future__ import annotations

import argparse
import sys

from .ingest import ingest_paths, save_chunks
from .pipeline import DEFAULT_DATA, DEFAULT_INDEX, RAG, build_index


def _filters(args) -> dict:
    f = {}
    if args.doc:
        f["doc_id"] = args.doc
    if args.kunde:
        f["kunde"] = args.kunde
    if args.maschine:
        f["maschinentyp"] = args.maschine
    if args.produkt:
        f["produkt"] = args.produkt
    return f


def _print_sources(hits) -> None:
    print("\nQuellen:")
    for c, score in hits:
        print(f"  [{score:.4f}] {c.doc_id} · {c.title} · {c.metadata.get('kunde_name')} · S. {c.metadata.get('seiten')}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description="RAG über Verfahrensbeschreibungen")
    p.add_argument("--index", default=str(DEFAULT_INDEX), help="Index-Verzeichnis")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="PDFs einlesen und Index bauen")
    pi.add_argument("paths", nargs="*", default=[str(DEFAULT_DATA)])
    pi.add_argument("--embedder", default=None, choices=["auto", "tfidf", "st"])
    pi.add_argument("--model", default=None, help="sentence-transformers Modellname")

    for name in ("search", "ask"):
        sp = sub.add_parser(name)
        sp.add_argument("question")
        sp.add_argument("-k", type=int, default=6)
        sp.add_argument("--doc", help="Filter: Dokument-Nr., z. B. VB-2019-003")
        sp.add_argument("--kunde", help="Filter: Teil des Kundennamens")
        sp.add_argument("--maschine", help="Filter: Maschinentyp, z. B. FL248")
        sp.add_argument("--produkt", help="Filter: Produktname (Teilstring)")
        sp.add_argument("--no-auto-filter", action="store_true", help="Kunden-/Dokumenterkennung abschalten")

    sub.add_parser("chat", help="Interaktiver Frage-Antwort-Modus")
    pe = sub.add_parser("export", help="Chunks als JSONL exportieren")
    pe.add_argument("out")
    pe.add_argument("paths", nargs="*", default=[str(DEFAULT_DATA)])

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        kw = {}
        if args.embedder:
            kw["embedder"] = args.embedder
        if args.model:
            kw["model_name"] = args.model
        idx = build_index(args.paths, args.index, **kw)
        docs = len({c.doc_id for c in idx.chunks})
        print(f"Index gebaut: {docs} Dokumente, {len(idx.chunks)} Chunks, Embedder={idx.embedder.name} -> {args.index}")
        return 0

    if args.cmd == "export":
        chunks = ingest_paths(args.paths)
        save_chunks(chunks, args.out)
        print(f"{len(chunks)} Chunks -> {args.out}")
        return 0

    if args.cmd == "search":
        rag = RAG(args.index, generator="extractive")
        hits, used = rag.retrieve(args.question, args.k, _filters(args), not args.no_auto_filter)
        if used:
            print(f"Filter: {used}")
        for c, score in hits:
            print(f"\n=== [{score:.4f}] {c.chunk_id} · {c.title} · {c.metadata.get('kunde')}")
            print(c.text)
        return 0

    if args.cmd == "ask":
        rag = RAG(args.index)
        ans = rag.ask(args.question, args.k, _filters(args), not args.no_auto_filter)
        print(ans.answer)
        _print_sources(ans.sources)
        return 0

    if args.cmd == "chat":
        rag = RAG(args.index)
        print("Frage eingeben (leer = Ende).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            ans = rag.ask(q)
            print(ans.answer)
            _print_sources(ans.sources)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
