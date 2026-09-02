"""Сховище чанків і їхніх ембедінгів: один файл SQLite.

Навіщо SQLite, а не векторна база: для корпусу в сотні або тисячі чанків
повний перебір косинусної подібності в numpy займає мілісекунди, і вся
механіка лишається прозорою. Заміна на pgvector або Qdrant це окремий крок,
коли корпус виросте до сотень тисяч чанків.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .chunking import Chunk

DEFAULT_DB = Path("data/index.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    heading TEXT NOT NULL,
    clause TEXT,
    text TEXT NOT NULL,
    ord INTEGER NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _row(r) -> dict:
    return dict(id=r[0], source=r[1], heading=r[2], clause=r[3], text=r[4], order=r[5])


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    # --- запис -------------------------------------------------------------
    def reset(self) -> None:
        self.conn.executescript(
            "DELETE FROM embeddings; DELETE FROM chunks; DELETE FROM meta;"
        )
        self.conn.commit()

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors), "кількість чанків і векторів має збігатись"
        vectors = np.asarray(vectors, dtype=np.float32)
        cur = self.conn.cursor()
        for c, v in zip(chunks, vectors):
            cur.execute(
                "INSERT INTO chunks(source, heading, clause, text, ord, meta) VALUES (?,?,?,?,?,?)",
                (
                    c.source,
                    c.heading,
                    c.clause,
                    c.text,
                    c.order,
                    json.dumps(c.meta, ensure_ascii=False),
                ),
            )
            cur.execute(
                "INSERT INTO embeddings(chunk_id, dim, vec) VALUES (?,?,?)",
                (cur.lastrowid, int(v.shape[0]), v.tobytes()),
            )
        self.conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value)
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    # --- читання -----------------------------------------------------------
    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def all_chunks(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, source, heading, clause, text, ord FROM chunks ORDER BY id"
        ).fetchall()
        return [_row(r) for r in rows]

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Повертає (ids, матриця векторів N×dim) для повного перебору."""
        rows = self.conn.execute(
            "SELECT chunk_id, dim, vec FROM embeddings ORDER BY chunk_id"
        ).fetchall()
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        return ids, mat

    def get(self, chunk_id: int) -> dict | None:
        r = self.conn.execute(
            "SELECT id, source, heading, clause, text, ord FROM chunks WHERE id=?",
            (chunk_id,),
        ).fetchone()
        return _row(r) if r else None
