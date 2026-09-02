"""Нарізання документів на чанки (шматки) для індексації.

Ідея: чанк має бути достатньо малим, щоб ембедінг описував одну думку,
і достатньо великим, щоб модель мала контекст для відповіді. Тут це робиться
з урахуванням структури Markdown: заголовки задають «шлях» чанка
(розділ → пункт), і цей шлях додається в текст, який ембедимо.

Для ПДР кожен пункт «### п. 12.4» стає окремим чанком, бо саме на пункт
має посилатись відповідь. Довгі абзаци без заголовків групуються до
TARGET_CHARS символів; дуже довгі абзаци ріжуться по реченнях.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TARGET_CHARS = 900  # бажаний розмір чанка
MAX_CHARS = 1500  # жорстка стеля, далі ріжемо по реченнях
CLAUSE_HEADING_RE = re.compile(r"^п\.\s*(\d+\.\d+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Chunk:
    source: str  # файл-джерело, відносний шлях
    heading: str  # шлях заголовків: «12. Швидкість руху › п. 12.4»
    clause: str | None  # номер пункту, якщо є («12.4»)
    text: str  # сам текст чанка (без заголовка)
    order: int = 0  # порядок у документі
    meta: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Текст, який іде в ембедінг і в BM25: заголовок + тіло."""
        return f"{self.heading}\n{self.text}"


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Ріже дуже довгий абзац по реченнях, не перевищуючи max_chars."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for sent in SENTENCE_SPLIT_RE.split(text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            parts.append(cur.strip())
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur.strip())
    return parts


def chunk_markdown(md: str, source: str) -> list[Chunk]:
    """Розбиває Markdown-документ на чанки з урахуванням заголовків."""
    h1 = ""
    h_sub = ""  # поточний підзаголовок (## або ###)
    clause: str | None = None
    buf: list[str] = []
    chunks: list[Chunk] = []

    def heading_path() -> str:
        return " › ".join(p for p in (h1, h_sub) if p)

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        for piece in _split_long(text):
            chunks.append(Chunk(source, heading_path(), clause, piece, len(chunks)))

    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            flush()
            h1, h_sub, clause = line[2:].strip(), "", None
            continue
        if line.startswith("## ") or line.startswith("### "):
            flush()
            h_sub = line.lstrip("#").strip()
            m = CLAUSE_HEADING_RE.match(h_sub)
            clause = m.group(1) if m else None
            continue
        if not line.strip():
            # Порожній рядок = межа абзацу. Абзаци групуємо до TARGET_CHARS,
            # але пункт (clause) завжди лишається одним чанком, поки не перевищить MAX_CHARS.
            if buf and clause is None and sum(len(b) for b in buf) >= TARGET_CHARS:
                flush()
            elif buf and buf[-1] != "":
                buf.append("")
            continue
        buf.append(line.strip())
    flush()

    for c in chunks:
        c.text = re.sub(r"\n{3,}", "\n\n", c.text).strip()
    return [c for c in chunks if c.text]


def chunk_plain(text: str, source: str) -> list[Chunk]:
    """Для .txt і PDF без Markdown-розмітки: абзаци групуються до TARGET_CHARS."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf = ""

    def flush(b: str) -> None:
        for piece in _split_long(b):
            chunks.append(Chunk(source, Path(source).stem, None, piece, len(chunks)))

    for p in paras:
        if buf and len(buf) + len(p) > TARGET_CHARS:
            flush(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}".strip()
    if buf:
        flush(buf)
    return chunks
