"""
Translator protocol and HuggingFace implementation.

Translators handle single-chunk translation only. Chunking, retry logic,
and orchestration are handled by the orchestration module.
"""

from typing import Protocol

from transformers import PreTrainedTokenizerBase

from .chunking import MAX_CHUNK_TOKENS, count_tokens
from .utils import TranslationError


class Translator(Protocol):
    """Protocol for translation backends."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single chunk of text."""
        ...

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        """Translate multiple chunks. Default loops over translate()."""
        ...


class HuggingFaceTranslator:
    """TranslateGemma using Hugging Face Transformers backend."""

    def __init__(
        self,
        model_name: str = "google/translategemma-12b-it",
        tokenizer: PreTrainedTokenizerBase | None = None,
        max_chunk_tokens: int = MAX_CHUNK_TOKENS,
        batch_size: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens
        self.batch_size = batch_size

        import torch
        from transformers import pipeline

        print(f"Loading model: {model_name}...")
        print("(This may take a few minutes on first run as the model downloads)")

        self.pipe = pipeline(
            "image-text-to-text",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded!")

    def _build_messages(self, text: str, source_lang: str, target_lang: str) -> list:
        """Build the message payload for a single chunk."""
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": source_lang,
                "target_lang_code": target_lang,
                "text": text,
            }],
        }]

    def is_truncated(self, text: str) -> bool:
        """Check if output was likely truncated by hitting max_new_tokens."""
        return count_tokens(text, self.tokenizer) >= self.max_chunk_tokens

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate a single chunk of text.

        Raises:
            TranslationError: If translation returns empty or truncated output.
        """
        messages = self._build_messages(text, source_lang, target_lang)
        output = self.pipe(
            text=messages,
            do_sample=False,
            pad_token_id=1,
            max_new_tokens=self.max_chunk_tokens,
        )
        result = output[0]["generated_text"][-1]["content"]

        if not result or not result.strip():
            raise TranslationError("Translation returned empty result")

        if self.is_truncated(result):
            raise TranslationError(
                f"Output truncated (hit {self.max_chunk_tokens} token limit)"
            )

        return result

    def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        """
        Translate multiple chunks using batched inference.

        Raises:
            TranslationError: If any translation returns empty or truncated output.
        """
        results = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start:batch_start + self.batch_size]

            if len(texts) > 1:
                batch_end = batch_start + len(batch)
                print(f"    Translating chunks {batch_start + 1}-{batch_end}/{len(texts)}...")

            batch_messages = [self._build_messages(c, source_lang, target_lang) for c in batch]
            outputs = self.pipe(
                text=batch_messages,
                do_sample=False,
                batch_size=len(batch),
                pad_token_id=1,
                max_new_tokens=self.max_chunk_tokens,
            )

            for i, output in enumerate(outputs):
                result = output[0]["generated_text"][-1]["content"]
                if not result or not result.strip():
                    raise TranslationError(
                        f"Translation returned empty result for chunk {batch_start + i + 1}"
                    )
                if self.is_truncated(result):
                    raise TranslationError(
                        f"Output truncated for chunk {batch_start + i + 1} "
                        f"(hit {self.max_chunk_tokens} token limit)"
                    )
                results.append(result)

        return results
