from unittest.mock import MagicMock, patch

from app.summarizer import _split_text, summarize


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


def test_summarize_empty_text() -> None:
    result = summarize("   ")
    assert "пустой" in result.lower()


def test_split_text_short() -> None:
    assert _split_text("Короткий текст", chunk_size=100) == ["Короткий текст"]


def test_split_text_long() -> None:
    text = ("Абзац текста для проверки нарезки. " * 20).strip()
    chunks = _split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_summarize_single_chunk(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("Готовое резюме.")

    with patch("app.summarizer._get_client", return_value=mock_client):
        result = summarize("Длинный исходный текст для проверки.", mode="short")

    assert result == "Готовое резюме."
    assert mock_client.chat.completions.create.call_count == 1


def test_summarize_map_reduce(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _mock_response("Часть 1"),
        _mock_response("Часть 2"),
        _mock_response("Итоговый конспект"),
    ]

    with (
        patch("app.summarizer._get_client", return_value=mock_client),
        patch("app.summarizer._split_text", return_value=["chunk1", "chunk2"]),
    ):
        result = summarize("Очень длинный текст", mode="notes")

    assert result == "Итоговый конспект"
    assert mock_client.chat.completions.create.call_count == 3
