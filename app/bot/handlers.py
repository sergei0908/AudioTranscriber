import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message

from app.config import Settings

log = logging.getLogger(__name__)
router = Router()

HELP_TEXT = (
    "Пришлите голосовое, аудиофайл, видео или видеосообщение (кружок), "
    "и я пришлю краткое резюме и полную расшифровку.\n\n"
    "Ограничение Telegram: боты принимают файлы не больше 20 МБ."
)


@dataclass(frozen=True)
class MediaInfo:
    file_id: str
    size: int | None
    suffix: str


def extract_media(message: Message) -> MediaInfo | None:
    if message.voice:
        voice = message.voice
        return MediaInfo(voice.file_id, voice.file_size, ".ogg")
    if message.audio:
        audio = message.audio
        suffix = Path(audio.file_name or "").suffix or ".mp3"
        return MediaInfo(audio.file_id, audio.file_size, suffix)
    if message.video:
        video = message.video
        suffix = Path(video.file_name or "").suffix or ".mp4"
        return MediaInfo(video.file_id, video.file_size, suffix)
    if message.video_note:
        note = message.video_note
        return MediaInfo(note.file_id, note.file_size, ".mp4")
    doc = message.document
    if doc and (doc.mime_type or "").startswith(("audio/", "video/")):
        suffix = Path(doc.file_name or "").suffix or ".bin"
        return MediaInfo(doc.file_id, doc.file_size, suffix)
    return None


def has_media(message: Message) -> bool:
    return extract_media(message) is not None


@router.message(Command("start", "help"))
async def cmd_start(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(has_media)
async def on_media(message: Message, bot: Bot, settings: Settings) -> None:
    media = extract_media(message)
    if media is None:
        return

    if media.size and media.size > settings.max_file_bytes:
        await message.answer(
            f"Файл слишком большой: боты могут скачивать не больше {settings.max_file_mb} МБ. "
            "Пришлите файл поменьше или только звук."
        )
        return

    status = await message.answer("Скачиваю файл…")
    try:
        Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=settings.tmp_dir) as workdir:
            src = Path(workdir) / f"input{media.suffix}"
            await bot.download(media.file_id, destination=src)
            size_kb = src.stat().st_size // 1024
        await status.edit_text(f"Файл скачан: {size_kb} КБ, формат {media.suffix}")
    except TelegramBadRequest as exc:
        log.warning("Telegram отклонил запрос: %s", exc)
        await status.edit_text("Не удалось скачать файл. Возможно, он больше 20 МБ.")


@router.message()
async def fallback(message: Message) -> None:
    await message.answer("Пришлите голосовое, аудио или видео. Подсказка: /help")
