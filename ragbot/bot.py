"""Telegram-бот (aiogram 3): те саме, що CLI, але в чаті.

  BOT_TOKEN у .env → python -m ragbot.bot

Кожен чат має свою історію (останні HISTORY_TURNS ходів), тож можна
ставити уточнювальні запитання. Історія живе в пам'яті процесу: після
перезапуску вона порожня, для навчального бота цього досить.
"""

from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from .agent import ask
from .retriever import Retriever

load_dotenv()
HISTORY_TURNS = 6
logging.basicConfig(level=logging.INFO)

dp = Dispatcher()
retriever: Retriever | None = None
histories: dict[int, list] = {}


@dp.message(CommandStart())
async def start(m: Message) -> None:
    await m.answer(
        "Привіт. Я відповідаю на запитання за базою документів (зараз це Правила дорожнього руху "
        "України) і вказую пункт, на який спираюсь.\n\nСпитай, наприклад: "
        "«Яка максимальна швидкість у житловій зоні?»"
    )


@dp.message(F.text == "/reset")
async def reset(m: Message) -> None:
    histories.pop(m.chat.id, None)
    await m.answer("Історію діалогу очищено.")


@dp.message(F.text)
async def answer(m: Message) -> None:
    assert retriever is not None
    await m.bot.send_chat_action(m.chat.id, "typing")
    history = histories.get(m.chat.id)
    try:
        # Виклик моделі синхронний, тому виносимо його в окремий потік, щоб не блокувати бота.
        ans = await asyncio.to_thread(ask, m.text, retriever, history)
    except Exception as e:  # noqa: BLE001
        logging.exception("помилка відповіді")
        await m.answer(f"Не вдалося відповісти: {type(e).__name__}. Спробуй ще раз.")
        return
    # Історію тримаємо короткою: беремо повідомлення останніх ходів.
    histories[m.chat.id] = ans.messages[-HISTORY_TURNS * 2 :]
    text = ans.text.strip() + f"\n\n<i>{ans.summary_line()}</i>"
    await m.answer(text[:4000], parse_mode="HTML")


async def main() -> None:
    global retriever
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("BOT_TOKEN не заданий у .env")
    retriever = Retriever()
    bot = Bot(token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
