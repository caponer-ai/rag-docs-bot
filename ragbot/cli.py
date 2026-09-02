"""Консольний інтерфейс.

python -m ragbot.cli "Яка максимальна швидкість у населеному пункті?"
python -m ragbot.cli            # інтерактивний режим, діалог з пам'яттю
python -m ragbot.cli --search "обгін на переході"   # лише пошук, без моделі
"""

from __future__ import annotations

import argparse
import sys

from .retriever import Retriever


def print_hits(retriever: Retriever, query: str, k: int) -> None:
    for h in retriever.search(query, k=k):
        ranks = f"vec={h.vec_rank or '-'} bm25={h.bm25_rank or '-'}"
        print(f"\n[{h.cite}] {h.chunk['heading']}  ({ranks})\n{h.chunk['text'][:400]}")


def print_answer(ans) -> None:
    print("\n" + ans.text.strip())
    if ans.sources:
        print(
            "\nДжерела, які бачила модель: "
            + ", ".join(sorted({h.cite for h in ans.sources}))
        )
    print("· " + ans.summary_line())


def main() -> None:
    ap = argparse.ArgumentParser(description="Питання до бази документів")
    ap.add_argument("question", nargs="?", help="запитання; без нього стартує діалог")
    ap.add_argument("--search", metavar="ЗАПИТ", help="показати лише результати пошуку")
    ap.add_argument("-k", type=int, default=6)
    args = ap.parse_args()

    retriever = Retriever()
    if args.search:
        print_hits(retriever, args.search, args.k)
        return

    from .agent import ask  # імпорт тут, щоб --search працював без ключа API

    if args.question:
        print_answer(ask(args.question, retriever))
        return

    print("Діалог з базою документів. Порожній рядок або Ctrl+C для виходу.")
    history = None
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ans = ask(q, retriever, history)
        print_answer(ans)
        history = ans.messages


if __name__ == "__main__":
    sys.exit(main())
