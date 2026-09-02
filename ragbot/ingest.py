"""Індексація: документи з data/docs → чанки → ембедінги → SQLite.

Запуск:  python -m ragbot.ingest [--docs data/docs] [--db data/index.sqlite]
Підтримує .md, .txt, .pdf (текст витягується через pypdf).
Індекс перебудовується з нуля щоразу: для навчального проєкту це простіше
і надійніше за інкрементальні оновлення.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .chunking import Chunk, chunk_markdown, chunk_plain
from .embeddings import MODEL_NAME, embed_texts
from .store import DEFAULT_DB, Store


def load_documents(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(docs_dir).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".md":
            chunks += chunk_markdown(path.read_text(encoding="utf-8"), rel)
        elif suffix == ".txt":
            chunks += chunk_plain(path.read_text(encoding="utf-8"), rel)
        elif suffix == ".pdf":
            from pypdf import PdfReader

            text = "\n\n".join(
                (p.extract_text() or "") for p in PdfReader(str(path)).pages
            )
            chunks += chunk_plain(text, rel)
    return chunks


def build_index(docs_dir: Path, db_path: Path) -> int:
    t0 = time.time()
    chunks = load_documents(docs_dir)
    if not chunks:
        raise SystemExit(f"У {docs_dir} не знайдено документів (.md/.txt/.pdf)")
    n_files = len({c.source for c in chunks})
    print(
        f"чанків: {len(chunks)} з {n_files} файлів; ембедимо моделлю {MODEL_NAME} ..."
    )
    vectors = embed_texts([c.embed_text for c in chunks])
    store = Store(db_path)
    store.reset()
    store.add_chunks(chunks, vectors)
    store.set_meta("embed_model", MODEL_NAME)
    store.set_meta("dim", str(vectors.shape[1]))
    sizes = sorted(len(c.text) for c in chunks)
    print(
        f"готово за {time.time() - t0:.1f} с: {store.count()} чанків, dim={vectors.shape[1]}, "
        f"довжина чанка медіана {sizes[len(sizes) // 2]} симв., макс {sizes[-1]}"
    )
    return store.count()


def main() -> None:
    ap = argparse.ArgumentParser(description="Побудувати індекс з папки документів")
    ap.add_argument("--docs", default="data/docs", type=Path)
    ap.add_argument("--db", default=DEFAULT_DB, type=Path)
    args = ap.parse_args()
    build_index(args.docs, args.db)


if __name__ == "__main__":
    main()
