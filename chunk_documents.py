#!/usr/bin/env python3
"""
Chunk documents into token-sized pieces for inspection.

Splits .txt files into chunks using the same token-based chunking logic
as translate_documents.py, writing each chunk as a separate file.
No translation or model loading required — only the tokenizer.

Usage:
    uv run python chunk_documents.py ./my_documents
    uv run python chunk_documents.py ./my_documents --output-dir ./chunks --chunk-size 500
"""

import argparse
import sys
from pathlib import Path

from translate_documents import (
    MAX_CHUNK_TOKENS,
    chunk_text_by_tokens,
    count_tokens,
    detect_language,
    get_language_name,
    load_tokenizer,
)


def main():
    parser = argparse.ArgumentParser(
        description="Chunk documents into token-sized text files",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .txt files to chunk",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: input_dir/chunks)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="en",
        help="Target language code; files detected as this language are skipped (default: en)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=MAX_CHUNK_TOKENS,
        help=f"Maximum tokens per chunk (default: {MAX_CHUNK_TOKENS}, max: {MAX_CHUNK_TOKENS})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/translategemma-12b-it",
        help="HF model to load the tokenizer from (default: google/translategemma-12b-it)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded HuggingFace models (default: HF_HOME env var)",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        default=False,
        help="Allow downloading models from HuggingFace Hub (default: offline mode, models must already be cached)",
    )

    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a valid directory")
        sys.exit(1)

    if args.chunk_size > MAX_CHUNK_TOKENS:
        print(f"Warning: --chunk-size {args.chunk_size} exceeds maximum of {MAX_CHUNK_TOKENS}, clamping")
        args.chunk_size = MAX_CHUNK_TOKENS

    output_dir = args.output_dir or (args.input_dir / "chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model, cache_dir=args.cache_dir, fetch=args.fetch)

    txt_files = sorted(args.input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(txt_files)} file(s), chunk size: {args.chunk_size} tokens")
    print(f"Output directory: {output_dir}")

    for txt_file in txt_files:
        print(f"\n{'='*60}")
        print(f"Processing: {txt_file.name}")

        content = None
        for encoding in ["utf-8", "utf-8-sig", "cp1252", "iso-8859-15", "latin-1"]:
            try:
                with open(txt_file, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"  Error: Could not decode file")
            continue

        if not content.strip():
            print(f"  Skipping empty file")
            continue

        # Detect language
        sample_start = max(0, len(content) // 4)
        sample = content[sample_start:sample_start + 1000]
        detected_lang = detect_language(sample) or detect_language(content[:1000])

        if detected_lang is None:
            print(f"  Error: Could not detect language, skipping")
            continue

        print(f"  Detected language: {get_language_name(detected_lang)} ({detected_lang})")

        if detected_lang == args.target_lang:
            print(f"  Skipping: already in {get_language_name(args.target_lang)}")
            continue

        # Chunk
        total_tokens = count_tokens(content, tokenizer)
        print(f"  Total tokens: {total_tokens}")

        if total_tokens <= args.chunk_size:
            chunks = [content]
        else:
            chunks = chunk_text_by_tokens(content, tokenizer, max_tokens=args.chunk_size)

        print(f"  Split into {len(chunks)} chunk(s)")

        # Write chunks
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = count_tokens(chunk, tokenizer)
            chunk_path = output_dir / f"{txt_file.stem}_chunk{i:03d}.txt"
            with open(chunk_path, "w", encoding="utf-8") as f:
                f.write(chunk)
            print(f"  Wrote {chunk_path.name} ({chunk_tokens} tokens)")

    print(f"\n{'='*60}")
    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
