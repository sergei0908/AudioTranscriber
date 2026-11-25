import asyncio
import logging

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from app.bot.handlers import router
from app.config import get_settings


async def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()
    dp["settings"] = settings
    dp.include_router(router)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
