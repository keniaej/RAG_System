"""Antwortgenerierung.

* ``ClaudeGenerator`` – formuliert die Antwort mit Claude (Anthropic API) strikt
  auf Basis der gefundenen Chunks und zitiert die Quellen ``[VB-…]``.
* ``ExtractiveGenerator`` – Fallback ohne API-Key: gibt die relevantesten
  Passagen mit Quelle zurück.
"""
from __future__ import annotations

import os

from .ingest import Chunk

SYSTEM_PROMPT = """Du bist ein Assistent für die Anwendungstechnik und beantwortest Fragen zu \
Verfahrensbeschreibungen von KS Feinstzerkleinerern (Maschinenkonfiguration, gefahrene Produkte, \
Schneidsatzbestückung, Prozessparameter).

Regeln:
- Antworte ausschließlich auf Basis der bereitgestellten Quellen. Erfinde keine Werte.
- Nenne Zahlen exakt mit Einheit, so wie sie in der Quelle stehen.
- Zitiere nach jeder Aussage die Quelle in eckigen Klammern, z. B. [VB-2019-003, Formfleisch-Emulsion Geflügel].
- Ist ein Wert in der Quelle als "k. A." oder "nicht erfasst" angegeben, sag das ausdrücklich.
- Wenn die Quellen die Frage nicht beantworten, sag klar, dass die Information nicht vorliegt.
- Bei Vergleichen über mehrere Kunden: nutze eine kurze Tabelle.
- Antworte in der Sprache der Frage (standardmäßig Deutsch), knapp und sachlich."""


def format_context(hits: list[tuple[Chunk, float]]) -> str:
    parts = []
    for n, (c, _score) in enumerate(hits, start=1):
        label = f"{c.doc_id}, {c.title}"
        parts.append(f'<quelle nr="{n}" ref="{label}" seiten="{c.metadata.get("seiten")}">\n{c.text}\n</quelle>')
    return "\n\n".join(parts)


class ClaudeGenerator:
    def __init__(self, model: str | None = None, max_tokens: int = 1200) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("RAG_LLM_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def generate(self, question: str, hits: list[tuple[Chunk, float]]) -> str:
        if not hits:
            return "Zu dieser Frage wurden keine passenden Dokumentstellen gefunden."
        user = f"Quellen:\n\n{format_context(hits)}\n\nFrage: {question}"
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


class ExtractiveGenerator:
    """Ohne LLM: liefert die Top-Passagen als Antwort."""

    def __init__(self, max_passages: int = 3) -> None:
        self.max_passages = max_passages

    def generate(self, question: str, hits: list[tuple[Chunk, float]]) -> str:
        if not hits:
            return "Zu dieser Frage wurden keine passenden Dokumentstellen gefunden."
        out = ["(Extraktiver Modus – kein LLM konfiguriert. Relevanteste Passagen:)\n"]
        for c, _ in hits[: self.max_passages]:
            body = c.text.split("\n", 1)[1] if "\n" in c.text else c.text
            out.append(f"### [{c.doc_id}, {c.title}] – {c.metadata.get('kunde', '')}\n{body}\n")
        return "\n".join(out)


def make_generator(kind: str = "auto"):
    """``auto``: Claude wenn ``anthropic`` installiert und ANTHROPIC_API_KEY gesetzt ist."""
    if kind in ("auto", "claude"):
        try:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY nicht gesetzt")
            return ClaudeGenerator()
        except Exception as exc:
            if kind == "claude":
                raise
            print(f"[rag] LLM nicht verfügbar ({exc}) – nutze extraktiven Modus.")
    return ExtractiveGenerator()
