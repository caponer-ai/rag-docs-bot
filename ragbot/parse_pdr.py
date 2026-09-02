"""Перетворює HTML-сторінку ПДР з zakon.rada.gov.ua на Markdown-файли по розділах.

Вхід:  data/raw/pdr.html (версія «Текст для друку»).
Вихід: data/docs/pdr/NN-назва.md, один файл на розділ Правил.

Чому свій парсер, а не готова бібліотека: текст ПДР має чітку структуру
«розділ → пункт N.N.», і саме номер пункту потрібен у відповідях бота
(«п. 12.4»), тож зберігаємо його як заголовок третього рівня в Markdown.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

RAW = Path("data/raw/pdr.html")
OUT = Path("data/docs/pdr")

SECTION_RE = re.compile(r"^(\d{1,2})\.\s+([А-ЯІЇЄҐ][^a-z]{3,})$")  # «12. ШВИДКІСТЬ РУХУ»
CLAUSE_RE = re.compile(r"^(\d{1,2}\.\d{1,2})\.\s")                # «12.4. ...»
STOP_MARKERS = ("Про Правила дорожнього руху", "Постійна адреса:", "Публікації документа")


def html_to_lines(raw: str) -> list[str]:
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S)
    body = re.sub(r"</(p|div|h\d|li|tr)>", "\n", body, flags=re.I)
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "", body))
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def slug(title: str) -> str:
    t = title.lower()
    t = re.sub(r"[^а-яіїєґa-z0-9]+", "-", t).strip("-")
    return t[:40].rstrip("-")


def split_sections(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    """Повертає [(номер, назва, рядки)], лише для основного тексту Правил."""
    # Правила починаються з «1. ЗАГАЛЬНІ ПОЛОЖЕННЯ», все до цього = преамбула постанови.
    start = next(i for i, ln in enumerate(lines) if ln == "1. ЗАГАЛЬНІ ПОЛОЖЕННЯ")
    sections: list[tuple[str, str, list[str]]] = []
    cur: tuple[str, str, list[str]] | None = None
    expected = 1
    for ln in lines[start:]:
        m = SECTION_RE.match(ln)
        # Заголовком розділу вважаємо лише наступний за номером (у розділі 33 є підрозділи «1. Попереджувальні знаки»,
        # але вони не у верхньому регістрі повністю, тож регекс їх і так не ловить).
        if m and int(m.group(1)) == expected and len(ln) < 120:
            cur = (m.group(1), m.group(2).strip(), [])
            sections.append(cur)
            expected += 1
            continue
        if cur is None:
            continue
        # Кінець документа: службовий блок сайту після тексту.
        if ln.startswith(STOP_MARKERS) and expected > 30:
            break
        cur[2].append(ln)
    return sections


def section_to_markdown(num: str, title: str, body: list[str]) -> str:
    out = [f"# {num}. {title.capitalize() if title.isupper() else title}", ""]
    for ln in body:
        # Службові примітки виду «{Пункт 12.4 із змінами ... }» до пошуку не потрібні.
        if ln.startswith("{") and ln.endswith("}"):
            continue
        c = CLAUSE_RE.match(ln)
        if c:
            out.append(f"### п. {c.group(1)}")
            out.append("")
            out.append(ln)
            out.append("")
        else:
            out.append(ln)
            out.append("")
    return "\n".join(out).strip() + "\n"


def main() -> None:
    lines = html_to_lines(RAW.read_text(encoding="utf-8", errors="replace"))
    sections = split_sections(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.md"):
        old.unlink()
    total_clauses = 0
    for num, title, body in sections:
        md = section_to_markdown(num, title, body)
        total_clauses += sum(1 for ln in body if CLAUSE_RE.match(ln))
        (OUT / f"{int(num):02d}-{slug(title)}.md").write_text(md, encoding="utf-8")
    print(f"розділів: {len(sections)}, пунктів N.N.: {total_clauses}, файлів у {OUT}")


if __name__ == "__main__":
    main()
