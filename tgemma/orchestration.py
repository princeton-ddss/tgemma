"""
Orchestration for document translation.

Handles chunking, retry logic, merging, and file-level translation workflows.
"""

from collections.abc import Callable
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .chunking import chunk_text_by_tokens, count_tokens
from .detection import LANGUAGES, detect_language, get_language_name
from .translator import HuggingFaceTranslator
from .utils import SkippedFileError, TranslationError, read_file_with_fallback


def load_tokenizer(model_name: str, cache_dir: str | None = None) -> PreTrainedTokenizerBase:
    """
    Load a HuggingFace tokenizer from local cache.

    Args:
        model_name: HuggingFace model name.
        cache_dir: Optional cache directory override.

    Returns:
        Loaded tokenizer.

    Raises:
        OSError: If tokenizer not found in cache.
    """
    return AutoTokenizer.from_pretrained(model_name, local_files_only=True, cache_dir=cache_dir)


def download_tokenizer(model_name: str, cache_dir: str | None = None) -> None:
    """
    Download tokenizer files from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name.
        cache_dir: Optional cache directory override.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        model_name,
        allow_patterns=["tokenizer*", "special_tokens_map.json"],
        cache_dir=cache_dir,
    )


def get_output_path(input_path: Path, output_dir: Path, suffix: str, target_lang: str) -> Path:
    """Generate the output file path for a translated file."""
    return output_dir / f"{input_path.stem}{suffix.format(target_lang=target_lang)}.txt"


def translate_file(
    translator: HuggingFaceTranslator,
    input_path: Path,
    output_dir: Path,
    source_lang: str | None = None,
    target_lang: str = "en",
    suffix: str = "_translated_{target_lang}",
    force: bool = False,
    on_progress: Callable[..., None] = print,
) -> None:
    """
    Translate a single file.

    Args:
        translator: Translator instance.
        input_path: Path to input file.
        output_dir: Directory for output files.
        source_lang: Source language code (auto-detect if None).
        target_lang: Target language code.
        suffix: Output filename suffix template.
        force: Re-translate even if output exists.
        on_progress: Callback for progress messages.

    Raises:
        SkippedFileError: If file should be skipped.
        TranslationError: If translation fails.
    """
    on_progress(f"\n{'=' * 60}")
    on_progress(f"Processing: {input_path.name}")

    output_path = get_output_path(input_path, output_dir, suffix, target_lang)
    if output_path.exists() and not force:
        raise SkippedFileError(f"Output file already exists: {output_path.name}")

    content = read_file_with_fallback(input_path)

    if not content.strip():
        raise SkippedFileError("Empty file")

    on_progress(f"  File size: {len(content):,} characters")

    # Detect language
    detected_lang = source_lang
    if detected_lang is None:
        detected_lang = detect_language(content)
        if detected_lang is None:
            raise TranslationError("Could not detect language (text may be too short or mixed)")
        on_progress(f"  Detected language: {get_language_name(detected_lang)} ({detected_lang})")
    else:
        on_progress(f"  Source language: {get_language_name(detected_lang)} ({detected_lang})")

    if detected_lang == target_lang:
        raise SkippedFileError(f"Already in {get_language_name(target_lang)}")

    if detected_lang not in LANGUAGES:
        on_progress(f"  Note: '{detected_lang}' not in common language list, attempting anyway...")

    # Translate
    on_progress(f"  Translating to: {get_language_name(target_lang)} ({target_lang})...")
    translated = translate_text(content, translator, detected_lang, target_lang)

    # Sanity check
    if len(translated) > 100 and detected_lang != "en":
        if content[:100] == translated[:100]:
            on_progress("  Warning: Output appears identical to input - translation may have failed")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)

    on_progress(f"  Saved to: {output_path}")
    on_progress(f"  Output size: {len(translated):,} characters")


def translate_text(
    text: str,
    translator: HuggingFaceTranslator,
    source_lang: str,
    target_lang: str = "en",
    max_retries: int = 3,
) -> str:
    """
    Translate text, chunking if necessary.

    Handles the chunk -> translate -> merge flow, with retry logic for
    chunks that produce truncated output.

    Args:
        text: Full document text to translate.
        translator: Translator backend to use.
        source_lang: Source language code.
        target_lang: Target language code.
        max_retries: Maximum retry attempts for truncated chunks.

    Returns:
        Translated document text.
    """
    tokenizer = translator.tokenizer
    max_tokens = translator.max_chunk_tokens

    if count_tokens(text, tokenizer) <= max_tokens:
        return _translate_chunk_with_retry(
            text, translator, source_lang, target_lang, tokenizer, max_tokens, max_retries
        )

    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=max_tokens)
    print(f"    Document is long - splitting into {len(chunks)} chunks...")

    try:
        return "\n\n".join(translator.translate_batch(chunks, source_lang, target_lang))
    except TranslationError as e:
        if "truncated" not in str(e).lower():
            raise
        # A chunk produced truncated output — fall back to per-chunk retry for all chunks
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            result = _translate_chunk_with_retry(
                chunk,
                translator,
                source_lang,
                target_lang,
                tokenizer,
                max_tokens,
                max_retries,
                chunk_num=i + 1,
                total_chunks=len(chunks),
            )
            translated_chunks.append(result)
        return "\n\n".join(translated_chunks)


def _translate_chunk_with_retry(
    text: str,
    translator: HuggingFaceTranslator,
    source_lang: str,
    target_lang: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    max_retries: int,
    chunk_num: int | None = None,
    total_chunks: int | None = None,
    retry_depth: int = 0,
) -> str:
    """Translate a chunk, retrying with smaller sub-chunks if truncated."""
    try:
        return translator.translate(text, source_lang, target_lang)
    except TranslationError as e:
        if "truncated" not in str(e).lower():
            raise

        if retry_depth >= max_retries:
            raise TranslationError(
                f"Output still truncated after {max_retries} retry attempts. "
                f"Input may be too complex to translate within token limits."
            ) from e

        input_tokens = count_tokens(text, tokenizer)
        half = max(int(input_tokens * 0.6), 1)

        prefix = f"Chunk {chunk_num}/{total_chunks}: " if chunk_num else ""
        print(
            f"      {prefix}Output truncated, splitting ({input_tokens} tokens) "
            f"and retrying (attempt {retry_depth + 1}/{max_retries})..."
        )

        sub_chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=half)
        parts = []
        for j, sub in enumerate(sub_chunks, 1):
            print(f"      Sub-chunk {j}/{len(sub_chunks)} ({count_tokens(sub, tokenizer)} tokens)...")
            result = _translate_chunk_with_retry(
                sub,
                translator,
                source_lang,
                target_lang,
                tokenizer,
                max_tokens,
                max_retries,
                retry_depth=retry_depth + 1,
            )
            parts.append(result)

        return "\n\n".join(parts)
