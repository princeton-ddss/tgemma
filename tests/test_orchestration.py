"""Tests for the orchestration module, focused on retry logic."""

from unittest.mock import MagicMock

import pytest

from tgemma.orchestration import _translate_chunk_with_retry, translate_text
from tgemma.utils import TranslationError


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that counts whitespace-separated words as tokens."""
    tokenizer = MagicMock()

    def mock_encode(text, add_special_tokens=True):
        return list(range(len(text.split())))

    def mock_decode(token_ids):
        return f"[{len(token_ids)} tokens]"

    tokenizer.encode = mock_encode
    tokenizer.decode = mock_decode
    return tokenizer


@pytest.fixture
def mock_translator(mock_tokenizer):
    translator = MagicMock()
    translator.tokenizer = mock_tokenizer
    translator.max_chunk_tokens = 10
    translator.batch_size = 4
    return translator


class TestTranslateChunkWithRetry:
    def test_success_on_first_attempt(self, mock_tokenizer, mock_translator):
        """No truncation: translate() called once, result returned directly."""
        mock_translator.translate.return_value = "translated text"

        result = _translate_chunk_with_retry(
            "some text", mock_translator, "de", "en", mock_tokenizer, max_tokens=10, max_retries=3
        )

        assert result == "translated text"
        mock_translator.translate.assert_called_once_with("some text", "de", "en")

    def test_truncation_triggers_sub_chunk_retry(self, mock_tokenizer, mock_translator):
        """Truncation on first call → text splits into sub-chunks → sub-chunks succeed."""
        mock_translator.translate.side_effect = [
            TranslationError("output truncated (hit 10 token limit)"),
            "first part",
            "second part",
        ]

        # 6 tokens total; 60% split → max_tokens=3; two 3-token paragraphs split cleanly
        text = "Word one two.\n\nWord three four."
        result = _translate_chunk_with_retry(
            text, mock_translator, "de", "en", mock_tokenizer, max_tokens=10, max_retries=3
        )

        assert result == "first part\n\nsecond part"
        assert mock_translator.translate.call_count == 3  # 1 failed + 2 sub-chunks

    def test_non_truncation_error_propagates_immediately(self, mock_tokenizer, mock_translator):
        """A non-truncation TranslationError must not trigger retry."""
        mock_translator.translate.side_effect = TranslationError("empty result")

        with pytest.raises(TranslationError, match="empty result"):
            _translate_chunk_with_retry(
                "some text", mock_translator, "de", "en", mock_tokenizer, max_tokens=10, max_retries=3
            )

        mock_translator.translate.assert_called_once()

    def test_exceeds_max_retries_raises(self, mock_tokenizer, mock_translator):
        """Persistent truncation after max_retries raises TranslationError."""
        mock_translator.translate.side_effect = TranslationError("output truncated")

        with pytest.raises(TranslationError, match="retry"):
            _translate_chunk_with_retry(
                "Word one two.\n\nWord three four.",
                mock_translator,
                "de",
                "en",
                mock_tokenizer,
                max_tokens=10,
                max_retries=1,
            )


class TestTranslateText:
    def test_short_text_uses_single_translate(self, mock_tokenizer, mock_translator):
        """Text within token limit bypasses translate_batch entirely."""
        mock_translator.translate.return_value = "translated"

        result = translate_text("short text", mock_translator, "de", "en")

        assert result == "translated"
        mock_translator.translate.assert_called_once()
        mock_translator.translate_batch.assert_not_called()

    def test_long_text_uses_translate_batch(self, mock_tokenizer, mock_translator):
        """Text exceeding token limit is chunked and sent to translate_batch."""
        mock_translator.max_chunk_tokens = 3
        mock_translator.translate_batch.return_value = ["chunk one", "chunk two", "chunk three"]

        # 9 words → 3 paragraphs of 3 tokens each; each splits at max_tokens=3
        text = "one two three.\n\nfour five six.\n\nseven eight nine."
        result = translate_text(text, mock_translator, "de", "en")

        assert result == "chunk one\n\nchunk two\n\nchunk three"
        mock_translator.translate_batch.assert_called_once()
        mock_translator.translate.assert_not_called()

    def test_batch_truncation_falls_back_to_per_chunk_retry(self, mock_tokenizer, mock_translator):
        """translate_batch truncation triggers per-chunk _translate_chunk_with_retry fallback."""
        mock_translator.max_chunk_tokens = 3
        mock_translator.translate_batch.side_effect = TranslationError("output truncated for chunk 2")
        mock_translator.translate.return_value = "retried chunk"

        text = "one two three.\n\nfour five six.\n\nseven eight nine."
        result = translate_text(text, mock_translator, "de", "en")

        assert result == "retried chunk\n\nretried chunk\n\nretried chunk"
        mock_translator.translate_batch.assert_called_once()
        assert mock_translator.translate.call_count == 3  # one per chunk

    def test_batch_non_truncation_error_propagates(self, mock_tokenizer, mock_translator):
        """Non-truncation error from translate_batch must propagate without retrying."""
        mock_translator.max_chunk_tokens = 3
        mock_translator.translate_batch.side_effect = TranslationError("empty result")

        text = "one two three.\n\nfour five six."
        with pytest.raises(TranslationError, match="empty result"):
            translate_text(text, mock_translator, "de", "en")

        mock_translator.translate.assert_not_called()
