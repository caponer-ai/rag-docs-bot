"""Агент, що відповідає по базі документів (Pydantic AI + Claude).

Схема «агентного RAG»: модель сама вирішує, коли і що шукати, і може
зробити кілька пошуків з різними формулюваннями, перш ніж відповісти.
Це відрізняється від «наївного RAG», де один пошук робиться до виклику
моделі і вона відповідає по тому, що знайшлось.

Інструмент один: search_docs(query, k). Він повертає модель фрагменти з
номерами пунктів, і інструкція вимагає посилатись на ці пункти у відповіді.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage

from .pricing import estimate_cost
from .retriever import Hit, Retriever

load_dotenv()

MODEL = os.getenv("MODEL", "claude-opus-5")
THINKING = os.getenv("THINKING", "medium")  # low | medium | high


def build_model():
    """Обирає провайдера за наявним ключем.

    ANTHROPIC_API_KEY → напряму в Anthropic (офіційний SDK).
    OPENROUTER_API_KEY → та сама модель через OpenRouter (один ключ на всі моделі,
    так налаштовані інші боти Влада). Ціни за токен у обох однакові.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(MODEL), {"thinking": THINKING, "max_tokens": 4000}
    if os.getenv("OPENROUTER_API_KEY"):
        from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

        settings = OpenRouterModelSettings(
            openrouter_reasoning={"effort": THINKING},
            openrouter_usage={"include": True},
            max_tokens=4000,
        )
        return OpenRouterModel(f"anthropic/{MODEL}"), settings
    raise SystemExit("Потрібен ANTHROPIC_API_KEY або OPENROUTER_API_KEY у .env (див. .env.example)")


INSTRUCTIONS = """Ти асистент, який відповідає на запитання СУВОРО за наданими документами.

Правила:
1. Перш ніж відповідати, знайди потрібні фрагменти інструментом search_docs. Якщо перший пошук не дав відповіді, переформулюй запит (інші слова, синоніми, номер пункту) і пошукай ще, до 3 разів.
2. Відповідай лише тим, що є у знайдених фрагментах. Не додавай знань «із загальної обізнаності». Якщо у документах відповіді немає, так і скажи: «У документах я цього не знайшов».
3. Після кожного твердження вказуй джерело у дужках у форматі (п. 12.4). Числа, строки й винятки цитуй точно, як у тексті.
4. Відповідай українською, стисло: спочатку пряма відповідь одним-двома реченнями, потім деталі й винятки, якщо вони є.
5. Якщо питання неоднозначне, назви варіанти тлумачення й відповідай на кожен.
"""


@dataclass
class Deps:
    """Залежності агента: пошуковик і журнал знайденого (для відображення джерел)."""

    retriever: Retriever
    seen: dict[int, Hit] = field(default_factory=dict)
    searches: list[str] = field(default_factory=list)


_model, _settings = build_model()
agent = Agent(_model, deps_type=Deps, instructions=INSTRUCTIONS, model_settings=_settings, retries=2)


@agent.tool
def search_docs(ctx: RunContext[Deps], query: str, k: int = 6) -> str:
    """Шукає у базі документів фрагменти, що відповідають запиту.

    Args:
        query: Пошуковий запит природною мовою або ключові слова (українською).
        k: Скільки фрагментів повернути (1..10), за замовчуванням 6.
    """
    k = max(1, min(int(k), 10))
    ctx.deps.searches.append(query)
    hits = ctx.deps.retriever.search(query, k=k)
    if not hits:
        return "Нічого не знайдено."
    parts = []
    for h in hits:
        ctx.deps.seen[h.chunk["id"]] = h
        parts.append(f"[{h.cite}] ({h.chunk['heading']})\n{h.chunk['text']}")
    return "\n\n---\n\n".join(parts)


@dataclass
class Answer:
    text: str
    sources: list[Hit]
    searches: list[str]
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    cost_usd: float | None
    messages: list[ModelMessage]
    model: str = MODEL

    def summary_line(self) -> str:
        cost = f"${self.cost_usd:.4f}" if self.cost_usd is not None else "n/a"
        return (
            f"{self.model} · пошуків: {len(self.searches)} · токени вхід {self.input_tokens} "
            f"(кеш {self.cache_read}) / вихід {self.output_tokens} · ~{cost}"
        )


def ask(
    question: str, retriever: Retriever, history: list[ModelMessage] | None = None
) -> Answer:
    """Одне запитання → відповідь з джерелами й витратами. history робить діалог багатоходовим."""
    deps = Deps(retriever=retriever)
    result = agent.run_sync(question, deps=deps, message_history=history or None)
    u = result.usage
    return Answer(
        text=result.output,
        sources=list(deps.seen.values()),
        searches=deps.searches,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read=u.cache_read_tokens,
        cache_write=u.cache_write_tokens,
        cost_usd=estimate_cost(
            MODEL,
            u.input_tokens,
            u.output_tokens,
            u.cache_read_tokens,
            u.cache_write_tokens,
        ),
        messages=result.all_messages(),
    )
