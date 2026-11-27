import logging
import os
import time
from typing import Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

log = logging.getLogger(__name__)

Mode = Literal["short", "bullets", "notes"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"
DEFAULT_TIMEOUT = 45.0
MAX_RETRIES = 3
CHUNK_SIZE = 12_000
CHUNK_OVERLAP = 500

BASE_RULES = (
    "Ты редактор и аналитик расшифровок аудио- и видеозаписей. "
    "Отвечай строго на том же языке, на котором написан исходный текст. "
    "Игнорируй слова-паразиты, оговорки и очевидные опечатки распознавания речи, "
    "не выдумывай факты, которых нет в исходнике."
)

MODE_PROMPTS: dict[str, str] = {
    "short": (
        f"{BASE_RULES}\n"
        "Сделай краткое резюме текста в 3–5 связных предложениях: "
        "только главная мысль, решения и итоги."
    ),
    "bullets": (
        f"{BASE_RULES}\n"
        "Выдели ключевые тезисы в виде маркированного списка (5–10 пунктов). "
        "Каждый пункт — одна чёткая мысль без воды."
    ),
    "notes": (
        f"{BASE_RULES}\n"
        "Составь структурированный конспект: тема, основные разделы с подзаголовками, "
        "ключевые факты, договорённости и задачи (если они упоминались)."
    ),
}

CHUNK_PROMPT = (
    f"{BASE_RULES}\n"
    "Это фрагмент длинной расшифровки. Сожми его, сохранив все важные факты, "
    "цифры, имена, решения и ключевые мысли."
)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Не задан OPENROUTER_API_KEY в переменных окружения.")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        timeout=float(os.getenv("OPENROUTER_TIMEOUT", str(DEFAULT_TIMEOUT))),
    )


def _split_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            split_pos = text.rfind("\n", start, end)
            if split_pos <= start:
                split_pos = text.rfind(". ", start, end)
                if split_pos > start:
                    split_pos += 1
            if split_pos <= start:
                split_pos = text.rfind(" ", start, end)
            if split_pos > start:
                end = split_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _call_llm(client: OpenAI, model: str, system_prompt: str, user_text: str) -> str:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            raise RuntimeError("Модель вернула пустой ответ.")
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            last_error = exc
            log.warning(
                "Попытка %d/%d не удалась: %s", attempt, MAX_RETRIES, type(exc).__name__
            )
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code < 500 and exc.status_code != 429:
                raise
            log.warning(
                "Попытка %d/%d: статус API %s", attempt, MAX_RETRIES, exc.status_code
            )

        if attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))

    assert last_error is not None
    raise last_error


def summarize(text: str, mode: str = "short") -> str:
    cleaned = text.strip()
    if not cleaned:
        return "Не удалось сделать резюме: текст для обработки пустой."

    if mode not in MODE_PROMPTS:
        log.warning("Неизвестный режим '%s', использую 'short'", mode)
        mode = "short"

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    started = time.perf_counter()

    try:
        client = _get_client()
        chunks = _split_text(cleaned)
        log.info(
            "Старт суммаризации: mode=%s, model=%s, chars=%d, chunks=%d",
            mode,
            model,
            len(cleaned),
            len(chunks),
        )

        if len(chunks) == 1:
            result = _call_llm(client, model, MODE_PROMPTS[mode], chunks[0])
        else:
            partials = [
                _call_llm(client, model, CHUNK_PROMPT, chunk) for chunk in chunks
            ]
            combined = "\n\n".join(partials)
            result = _call_llm(client, model, MODE_PROMPTS[mode], combined)

        elapsed = time.perf_counter() - started
        log.info(
            "Суммаризация завершена за %.2f с (out_chars=%d)", elapsed, len(result)
        )
        return result
    except RateLimitError:
        log.error("Превышен лимит запросов OpenRouter (429)")
        return "Сервис суммаризации перегружен (лимит запросов). Попробуйте повторить через минуту."
    except APITimeoutError:
        log.error("Таймаут при обращении к OpenRouter")
        return "Сервис суммаризации не ответил вовремя. Попробуйте позже."
    except APIStatusError as exc:
        log.error("Ошибка API OpenRouter: status=%s", exc.status_code)
        return "Ошибка внешнего сервиса суммаризации. Попробуйте позже."
    except Exception:
        log.exception("Непредвиденная ошибка при суммаризации")
        return "Не удалось обработать текст для резюме. Попробуйте позже."
