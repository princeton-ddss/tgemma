"""
Command-line interface for document translation.
"""

import traceback
from pathlib import Path
from typing import Annotated, Optional

import typer

from .chunking import MAX_CHUNK_TOKENS, chunk_text_by_tokens, count_tokens
from .detection import detect_language, get_language_name
from .extraction import SUPPORTED_EXTENSIONS, extract_text, get_supported_files
from .orchestration import (
    download_tokenizer,
    load_tokenizer,
    translate_file,
)
from .translator import HuggingFaceTranslator
from .utils import SkippedFileError, TranslationError

app = typer.Typer(
    help="Translate documents using TranslateGemma. Run 'tgemma <dir>' to translate.",
    invoke_without_command=True,
)


def get_tokenizer(model_name: str, fetch: bool):
    """Load tokenizer, downloading if needed and allowed."""
    try:
        return load_tokenizer(model_name)
    except OSError:
        if not fetch:
            print(f"Error: Tokenizer for '{model_name}' not found in cache.")
            print("  Run with --fetch to download, or manually with:")
            print(f"    huggingface-cli download {model_name} --include 'tokenizer*'")
            raise typer.Exit(1)
        print("Downloading tokenizer from HuggingFace Hub...")
        download_tokenizer(model_name)
        return load_tokenizer(model_name)


def run_translate(
    input_dir: Path,
    output_dir: Path | None,
    source_lang: str | None,
    target_lang: str,
    chunk_size: int,
    batch_size: int,
    model: str,
    suffix: str,
    fetch: bool,
    force: bool,
) -> None:
    """Core translate logic."""
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        raise typer.Exit(1)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        raise typer.Exit(1)

    if chunk_size > MAX_CHUNK_TOKENS:
        print(f"Warning: --chunk-size {chunk_size} exceeds maximum of {MAX_CHUNK_TOKENS}, clamping")
        chunk_size = MAX_CHUNK_TOKENS

    out_dir = output_dir or (input_dir / "translated")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TranslateGemma Document Translator")
    print("=" * 60)
    print(f"Chunk size: {chunk_size} tokens")

    tokenizer = get_tokenizer(model, fetch)
    print(f"Model: {model}")
    print("\nInitializing HuggingFace backend...")
    translator = HuggingFaceTranslator(
        model,
        tokenizer=tokenizer,
        max_chunk_tokens=chunk_size,
        batch_size=batch_size,
    )

    input_files = get_supported_files(input_dir)
    if not input_files:
        print(f"\nNo supported files found in {input_dir}")
        print(f"  Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        raise typer.Exit(1)

    print(f"\nFound {len(input_files)} file(s) to process")
    print(f"Output directory: {out_dir}")
    if force:
        print("Force mode: will re-translate existing files")

    success_count = 0
    skipped_count = 0
    failed_files = []

    for input_file in input_files:
        try:
            translate_file(translator, input_file, out_dir, source_lang, target_lang, suffix, force)
            success_count += 1
        except SkippedFileError as e:
            print(f"  Skipping: {e}")
            skipped_count += 1
        except TranslationError as e:
            print(f"  Error: {e}")
            failed_files.append(input_file.name)
        except Exception as e:
            print(f"  Error during translation: {e}")
            traceback.print_exc()
            failed_files.append(input_file.name)

    print(f"\n{'=' * 60}")
    print("Translation complete!")
    print(f"Successfully translated: {success_count}/{len(input_files)} files")
    if skipped_count:
        print(f"Skipped: {skipped_count} file(s)")
    if failed_files:
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"\nOutput directory: {out_dir}")
    print("=" * 60)


@app.callback()
def main_callback(
    ctx: typer.Context,
    input_dir: Annotated[Optional[Path], typer.Argument(help="Directory containing files to translate")] = None,
    output_dir: Annotated[Optional[Path], typer.Option(help="Output directory")] = None,
    source_lang: Annotated[Optional[str], typer.Option(help="Source language code (auto-detect)")] = None,
    target_lang: Annotated[str, typer.Option(help="Target language code")] = "en",
    chunk_size: Annotated[int, typer.Option(help="Maximum tokens per chunk")] = MAX_CHUNK_TOKENS,
    batch_size: Annotated[int, typer.Option(help="Chunks to translate in parallel")] = 1,
    model: Annotated[str, typer.Option(help="HuggingFace model name")] = "google/translategemma-12b-it",
    suffix: Annotated[str, typer.Option(help="Output filename suffix")] = "_translated_{target_lang}",
    fetch: Annotated[bool, typer.Option(help="Allow downloading from HuggingFace Hub")] = False,
    force: Annotated[bool, typer.Option(help="Re-translate even if output exists")] = False,
) -> None:
    """Translate documents using TranslateGemma."""
    if ctx.invoked_subcommand is None:
        if input_dir is None:
            print("Usage: tgemma <input_dir> [OPTIONS]")
            print("       tgemma chunk <input_dir> [OPTIONS]")
            print("\nRun 'tgemma --help' for more information.")
            raise typer.Exit(1)
        run_translate(
            input_dir, output_dir, source_lang, target_lang,
            chunk_size, batch_size, model, suffix, fetch, force,
        )


@app.command()
def chunk(
    input_dir: Annotated[Path, typer.Argument(help="Directory containing files to chunk")],
    output_dir: Annotated[Optional[Path], typer.Option(help="Output directory")] = None,
    target_lang: Annotated[str, typer.Option(help="Skip files already in this language")] = "en",
    chunk_size: Annotated[int, typer.Option(help="Maximum tokens per chunk")] = MAX_CHUNK_TOKENS,
    model: Annotated[str, typer.Option(help="HuggingFace model for tokenizer")] = "google/translategemma-12b-it",
    fetch: Annotated[bool, typer.Option(help="Allow downloading from HuggingFace Hub")] = False,
) -> None:
    """Split documents into token-sized chunks (no translation)."""
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        raise typer.Exit(1)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        raise typer.Exit(1)

    if chunk_size > MAX_CHUNK_TOKENS:
        print(f"Warning: --chunk-size {chunk_size} exceeds maximum of {MAX_CHUNK_TOKENS}, clamping")
        chunk_size = MAX_CHUNK_TOKENS

    out_dir = output_dir or (input_dir / "chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(model, fetch)

    input_files = get_supported_files(input_dir)
    if not input_files:
        print(f"No supported files found in {input_dir}")
        print(f"  Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        raise typer.Exit(1)

    print(f"Found {len(input_files)} file(s), chunk size: {chunk_size} tokens")
    print(f"Output directory: {out_dir}")

    for input_file in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file.name}")

        try:
            content = extract_text(input_file)
        except TranslationError as e:
            print(f"  Error: {e}")
            continue

        if not content.strip():
            print("  Skipping empty file")
            continue

        detected_lang = detect_language(content)
        if detected_lang is None:
            print("  Error: Could not detect language, skipping")
            continue

        print(f"  Detected language: {get_language_name(detected_lang)} ({detected_lang})")

        if detected_lang == target_lang:
            print(f"  Skipping: already in {get_language_name(target_lang)}")
            continue

        total_tokens = count_tokens(content, tokenizer)
        print(f"  Total tokens: {total_tokens}")

        if total_tokens <= chunk_size:
            chunks = [content]
        else:
            chunks = chunk_text_by_tokens(content, tokenizer, max_tokens=chunk_size)

        print(f"  Split into {len(chunks)} chunk(s)")

        for i, chunk_text in enumerate(chunks, 1):
            chunk_tokens = count_tokens(chunk_text, tokenizer)
            chunk_path = out_dir / f"{input_file.stem}_chunk{i:03d}.txt"
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk_text)
            print(f"  Wrote {chunk_path.name} ({chunk_tokens} tokens)")

    print(f"\n{'='*60}")
    print(f"Done. Output directory: {out_dir}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
