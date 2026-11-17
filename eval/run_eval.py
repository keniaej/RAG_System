"""Retrieval-Evaluation: Hit@k und MRR auf eval/questions.jsonl.

    python eval/run_eval.py [--index index] [-k 6]

Ein Treffer zählt, wenn einer der erwarteten Chunks (``expected``) unter den
Top-k ist. Bei Fragen mit mehreren erwarteten Chunks reicht einer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline import RAG  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / "index"))
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--no-auto-filter", action="store_true")
    args = ap.parse_args()

    rag = RAG(args.index, generator="extractive")
    rows = [json.loads(l) for l in Path(args.questions).read_text(encoding="utf-8").splitlines() if l.strip()]
    hit1 = hit3 = hitk = 0
    mrr = 0.0
    for row in rows:
        hits, used = rag.retrieve(row["question"], k=args.k, auto_filter=not args.no_auto_filter)
        ids = [c.chunk_id for c, _ in hits]
        rank = next((i + 1 for i, cid in enumerate(ids) if cid in row["expected"]), None)
        hit1 += rank == 1
        hit3 += bool(rank and rank <= 3)
        hitk += bool(rank)
        mrr += 1 / rank if rank else 0
        mark = "✓" if rank else "✗"
        print(f"{mark} rank={rank or '-':<2} {row['question'][:80]}" + (f"  filter={used}" if used else ""))
        if not rank:
            print(f"     erwartet {row['expected']}, gefunden {ids[:3]}")
    n = len(rows)
    print(f"\n{n} Fragen | Hit@1 {hit1/n:.0%} | Hit@3 {hit3/n:.0%} | Hit@{args.k} {hitk/n:.0%} | MRR {mrr/n:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
