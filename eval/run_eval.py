"""Оцінка якості на еталонному наборі eval/golden.jsonl.

Два рівні, бо RAG ламається у двох різних місцях:

1. Пошук (безкоштовно, без моделі): чи потрапив потрібний пункт у топ-k.
   Метрики hit@k і MRR, окремо для vector, bm25 і hybrid, щоб бачити,
   що саме дає гібрид.
2. Відповідь (платно, викликає модель): чи є у відповіді обов'язкові
   фрагменти (must) і посилання на потрібний пункт. Для питання поза
   документами must = «не знайшов», тобто перевіряємо чесну відмову.

Запуск:
  python -m eval.run_eval                # лише пошук
  python -m eval.run_eval --answers      # плюс відповіді моделі (витрати!)
Результат: eval/results.md (перезаписується) + eval/results_raw/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path

from ragbot.retriever import Retriever

ROOT = Path(__file__).parent
GOLDEN = ROOT / "golden.jsonl"
K = 5


def load_golden() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def eval_retrieval(retriever: Retriever, items: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with_clause = [it for it in items if it["clause"]]
    for mode in ("vector", "bm25", "hybrid"):
        hits, rr, misses = 0, 0.0, []
        for it in with_clause:
            cites = [
                h.chunk["clause"] for h in retriever.search(it["q"], k=K, mode=mode)
            ]
            if it["clause"] in cites:
                rank = cites.index(it["clause"]) + 1
                hits += 1
                rr += 1 / rank
            else:
                misses.append(
                    f"{it['clause']} ← {it['q'][:50]}… (топ: {', '.join(c or '-' for c in cites)})"
                )
        n = len(with_clause)
        out[mode] = {
            "n": n,
            "hit": hits,
            "hit_at_k": hits / n,
            "mrr": rr / n,
            "misses": misses,
        }
    return out


def eval_answers(retriever: Retriever, items: list[dict]) -> list[dict]:
    from ragbot.agent import ask

    rows = []
    for it in items:
        t0 = time.time()
        ans = ask(it["q"], retriever)
        text = ans.text.lower()
        must_ok = all(m.lower() in text for m in it["must"])
        cite_ok = (
            True
            if it["clause"] is None
            else bool(re.search(rf"п\.\s*{re.escape(it['clause'])}\b", ans.text))
        )
        rows.append(
            {
                "q": it["q"],
                "clause": it["clause"],
                "must_ok": must_ok,
                "cite_ok": cite_ok,
                "searches": len(ans.searches),
                "in": ans.input_tokens,
                "out": ans.output_tokens,
                "cache_read": ans.cache_read,
                "cost": ans.cost_usd,
                "sec": round(time.time() - t0, 1),
                "answer": ans.text,
            }
        )
        print(
            f"{'ok ' if must_ok and cite_ok else 'FAIL'} {it['q'][:60]}  ${ans.cost_usd} {rows[-1]['sec']}s"
        )
    return rows


def write_report(
    items: list[dict], retr: dict, answers: list[dict] | None, model: str | None
) -> None:
    n_total = len(items)
    lines = [f"# Результати оцінки ({date.today().isoformat()})", ""]
    lines += [
        f"Еталон: {n_total} питань, ручна розмітка (пункт ПДР + обов'язкові фрагменти відповіді), "
        f"{sum(1 for i in items if i['clause'])} з відповіддю в документах і "
        f"{sum(1 for i in items if not i['clause'])} поза документами. Один прогін.",
        "",
        f"## Пошук (top-{K})",
        "",
        "| режим | hit@5 | MRR |",
        "|---|---|---|",
    ]
    for mode, r in retr.items():
        lines.append(
            f"| {mode} | {r['hit']}/{r['n']} ({r['hit_at_k']:.0%}) | {r['mrr']:.2f} |"
        )
    for mode, r in retr.items():
        if r["misses"]:
            lines += ["", f"Промахи {mode}:", ""] + [f"- {m}" for m in r["misses"]]
    if answers:
        ok = sum(1 for a in answers if a["must_ok"] and a["cite_ok"])
        cost = sum(a["cost"] or 0 for a in answers)
        lines += [
            "",
            f"## Відповіді моделі ({model})",
            "",
            f"Правильних (є обов'язкові фрагменти і посилання на пункт): {ok}/{len(answers)}. "
            f"Сумарна оцінка вартості {cost:.3f} $, у середньому {cost / len(answers):.4f} $ на питання, "
            f"пошуків на питання {sum(a['searches'] for a in answers) / len(answers):.1f}, "
            f"час {sum(a['sec'] for a in answers) / len(answers):.0f} с.",
            "",
            "| # | питання | пункт | фрагменти | посилання | пошуків | $ |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, a in enumerate(answers, 1):
            lines.append(
                f"| {i} | {a['q'][:55]} | {a['clause'] or 'нема'} | {'так' if a['must_ok'] else 'НІ'} | "
                f"{'так' if a['cite_ok'] else 'НІ'} | {a['searches']} | {a['cost']} |"
            )
    (ROOT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nзвіт: {ROOT / 'results.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--answers",
        action="store_true",
        help="також прогнати відповіді моделі (платно)",
    )
    args = ap.parse_args()
    items = load_golden()
    retriever = Retriever()
    retr = eval_retrieval(retriever, items)
    for mode, r in retr.items():
        print(f"{mode:7} hit@{K} {r['hit']}/{r['n']}  MRR {r['mrr']:.2f}")
    answers, model = None, None
    if args.answers:
        from ragbot.agent import MODEL

        model = MODEL
        answers = eval_answers(retriever, items)
        raw = ROOT / "results_raw"
        raw.mkdir(exist_ok=True)
        (raw / f"{date.today().isoformat()}_{model}.jsonl").write_text(
            "\n".join(json.dumps(a, ensure_ascii=False) for a in answers),
            encoding="utf-8",
        )
    write_report(items, retr, answers, model)


if __name__ == "__main__":
    main()
