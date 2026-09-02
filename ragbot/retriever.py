"""Гібридний пошук: вектори (косинус) + BM25 (ключові слова) → злиття рангів.

Чому гібрид: маленька мультимовна модель ембедінгів плутає близькі за
змістом пункти (наприклад, різні обмеження швидкості), а BM25 точно ловить
числа й терміни («50 км/год», «автомагістраль»), зате не розуміє синонімів.
Reciprocal Rank Fusion (RRF) зводить два ранжування без підбору ваг:
score = сума 1 / (k + rank) по всіх ранжуваннях, k = 60 за статтею
Cormack, Clarke, Buettcher (2009).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import embed_query
from .store import DEFAULT_DB, Store

TOKEN_RE = re.compile(r"[а-яіїєґa-z0-9]+(?:[.,][0-9]+)?", re.I)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Проста токенізація для BM25: нижній регістр, слова й числа.

    Легкий «стемінг»: довгі слова обрізаємо до перших 6 літер, щоб
    «перехрестя», «перехресті», «перехрестям» зводились до одного токена.
    Це груба евристика (чергування на кшталт «швидкість/швидкості» вона
    не ловить), але для української без окремого стемера вона помітно
    піднімає збіг запиту з текстом.
    """
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if t.isalpha() and len(t) > 6:
            t = t[:6]
        out.append(t)
    return out


@dataclass
class Hit:
    chunk: dict
    score: float
    vec_rank: int | None
    bm25_rank: int | None

    @property
    def cite(self) -> str:
        c = self.chunk
        return f"п. {c['clause']}" if c["clause"] else c["heading"]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: список ранжувань (списки id) → зведений бал."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.store = Store(db_path)
        self.chunks = {c["id"]: c for c in self.store.all_chunks()}
        if not self.chunks:
            raise RuntimeError(
                f"Індекс порожній: спершу python -m ragbot.ingest ({db_path})"
            )
        self.ids, self.matrix = self.store.matrix()
        self.bm25_ids = list(self.chunks.keys())
        self.bm25 = BM25Okapi(
            [tokenize(f"{c['heading']}\n{c['text']}") for c in self.chunks.values()]
        )

    def vector_search(self, query: str, top: int) -> list[int]:
        q = embed_query(query)
        sims = self.matrix @ q  # вектори нормовані, тому це косинус
        order = np.argsort(-sims)[:top]
        return [int(self.ids[i]) for i in order]

    def bm25_search(self, query: str, top: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:top]
        return [self.bm25_ids[i] for i in order if scores[i] > 0]

    def search(
        self, query: str, k: int = 6, candidates: int = 20, mode: str = "hybrid"
    ) -> list[Hit]:
        """Пошук k найкращих чанків. mode: hybrid (RRF), vector або bm25 (для порівняння в eval)."""
        vec = (
            self.vector_search(query, candidates)
            if mode in ("hybrid", "vector")
            else []
        )
        bm = self.bm25_search(query, candidates) if mode in ("hybrid", "bm25") else []
        fused = rrf([r for r in (vec, bm) if r])
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            Hit(
                chunk=self.chunks[cid],
                score=score,
                vec_rank=vec.index(cid) + 1 if cid in vec else None,
                bm25_rank=bm.index(cid) + 1 if cid in bm else None,
            )
            for cid, score in top
        ]
