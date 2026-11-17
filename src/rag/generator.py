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


class OllamaGenerator:
    """Lokales LLM über Ollama (z. B. ``llama3.2:3b``) – kein API-Key, keine Daten verlassen den Rechner.

    Nutzt die HTTP-API von Ollama (Standard: http://localhost:11434), daher kein Zusatzpaket nötig.
    """

    def __init__(self, model: str | None = None, host: str | None = None, timeout: int = 300) -> None:
        self.model = model or os.getenv("RAG_OLLAMA_MODEL", "llama3.2:3b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout
        self._check()

    def _request(self, path: str, payload: dict | None = None, timeout: int = 5) -> dict:
        import json
        import urllib.request

        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _check(self) -> None:
        try:
            tags = self._request("/api/tags")
        except Exception as exc:
            raise RuntimeError(
                f"Ollama unter {self.host} nicht erreichbar – läuft die Ollama-App bzw. `ollama serve`? ({exc})"
            ) from exc
        names = {m.get("name", "") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise RuntimeError(
                f"Modell '{self.model}' nicht in Ollama installiert. Vorhanden: {sorted(names) or 'keine'}. "
                f"Mit `ollama pull {self.model}` laden oder RAG_OLLAMA_MODEL anpassen."
            )

    def generate(self, question: str, hits: list[tuple[Chunk, float]]) -> str:
        if not hits:
            return "Zu dieser Frage wurden keine passenden Dokumentstellen gefunden."
        user = f"Quellen:\n\n{format_context(hits)}\n\nFrage: {question}\n\nAntworte auf Deutsch und nenne die Quellen."
        resp = self._request(
            "/api/chat",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # niedrige Temperatur = weniger "Kreativität" bei Zahlen; großer Kontext für k Chunks
                "options": {"temperature": 0.1, "num_ctx": 8192},
            },
            timeout=self.timeout,
        )
        return resp.get("message", {}).get("content", "").strip()


def make_generator(kind: str | None = None):
    """Wählt das LLM. ``kind`` bzw. RAG_LLM_BACKEND: auto | claude | ollama | extractive.

    ``auto``: Claude, wenn ANTHROPIC_API_KEY gesetzt ist; sonst Ollama, wenn erreichbar;
    sonst extraktiver Modus.
    """
    kind = kind or os.getenv("RAG_LLM_BACKEND", "auto")
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "ollama":
        return OllamaGenerator()
    if kind == "claude":
        return ClaudeGenerator()
    # auto
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return ClaudeGenerator()
        except Exception as exc:
            print(f"[rag] Claude nicht verfügbar ({exc}).")
    try:
        return OllamaGenerator()
    except Exception as exc:
        print(f"[rag] Ollama nicht verfügbar ({exc}).")
    print("[rag] Kein LLM verfügbar – nutze extraktiven Modus.")
    return ExtractiveGenerator()
