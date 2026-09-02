"""Оцінка вартості виклику за токенами.

Ціни в доларах за 1 млн токенів, з прайсу Anthropic станом на 06.2026.
Кеш: читання 0.1 від ціни входу, запис 1.25 від ціни входу.
Це оцінка, а не рахунок: точну суму показує консоль Anthropic.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {  # model → (input, output)
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float | None:
    """Вартість у доларах або None, якщо модель не в таблиці.

    input_tokens тут вже включає cache_read і cache_write (так рахує Pydantic AI),
    тому кешовані токени віднімаємо від повної ціни і додаємо за своїм тарифом.
    """
    price = PRICES.get(model)
    if price is None:
        return None
    p_in, p_out = price
    plain = max(input_tokens - cache_read - cache_write, 0)
    usd = (
        plain * p_in
        + cache_read * p_in * 0.1
        + cache_write * p_in * 1.25
        + output_tokens * p_out
    ) / 1e6
    return round(usd, 5)
