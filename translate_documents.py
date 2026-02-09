#!/usr/bin/env python3
"""
TranslateGemma Document Translation Script
==========================================
Translates .txt files from various languages to English using Google's TranslateGemma model.
Supports automatic language detection and chunking for small context window.

Requirements:
- Ollama installed and running with translategemma model pulled
- OR Hugging Face Transformers with model downloaded

Usage with Ollama (recommended):
    uv run python translate_documents.py ./my_documents

Usage with Hugging Face:
    uv run python translate_documents.py ./my_documents --backend huggingface
"""

import argparse
import os
import sys
import re
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# HuggingFace reads HF_HOME / HF_HUB_CACHE at import time, so we must
# set them from --cache-dir before importing transformers or huggingface_hub.
# Similarly, set HF_HUB_OFFLINE=1 unless --fetch is present.
if "--cache-dir" in sys.argv:
    _idx = sys.argv.index("--cache-dir")
    if _idx + 1 < len(sys.argv):
        _hf_home = str(Path(sys.argv[_idx + 1]).resolve())
        os.environ["HF_HOME"] = _hf_home
        os.environ["HF_HUB_CACHE"] = os.path.join(_hf_home, "hub")
if "--fetch" not in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"

# Language detection
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Tokenizer
from transformers import AutoTokenizer

# Suppress noisy transformers log messages (tokenizer regex, pad_token_id, etc.)
# Actual errors (ERROR level) still come through.
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*pipelines sequentially on GPU.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

# Set seed for consistent language detection
DetectorFactory.seed = 0

# TranslateGemma context window is 2048 tokens (input + output).
# Reserve ~248 tokens for the prompt template and split the rest
# evenly between the input chunk and the generated translation.
MAX_CHUNK_TOKENS = 900

# ISO 639-1 language code to full name mapping
LANGUAGE_NAMES = {
    "es": "Spanish",
    "de": "German", 
    "fr": "French",
    "pt": "Portuguese",
    "cs": "Czech",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "hu": "Hungarian",
    "he": "Hebrew",
    "it": "Italian",
    "no": "Norwegian",
    "nb": "Norwegian",
    "nn": "Norwegian",
    "pl": "Polish",
    "sk": "Slovak",
    "sv": "Swedish",
    "tr": "Turkish",
    "ko": "Korean",
    "en": "English",
    "ca": "Catalan",
    "eu": "Basque"
}


def detect_language(text: str) -> Optional[str]:
    """Detect the language of the given text."""
    try:
        # Need enough text for reliable detection
        if len(text.strip()) < 20:
            return None
        lang_code = detect(text)
        return lang_code
    except LangDetectException:
        return None


def get_language_name(code: str) -> str:
    """Get the full language name from ISO code."""
    return LANGUAGE_NAMES.get(code, code.capitalize())


def load_tokenizer(model_name: str, fetch: bool = False):
    """Load a HuggingFace tokenizer, downloading only tokenizer files if needed.

    Tries loading from local cache first. If not found and fetch=True,
    downloads just the tokenizer files (not the full model weights).
    Cache location is controlled by the HF_HOME env var (set via --cache-dir).
    """
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        if not fetch:
            print(f"Error: Tokenizer for '{model_name}' not found in cache.")
            print(f"  Run with --fetch on a node with internet access to download it,")
            print(f"  or download it manually with:")
            print(f"    uv run hf download {model_name} --include 'tokenizer*' 'special_tokens_map.json'")
            sys.exit(1)
        print(f"Tokenizer not cached locally, downloading from HuggingFace Hub...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            model_name,
            allow_patterns=["tokenizer*", "special_tokens_map.json"],
        )
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[str]:
    """
    Split text into chunks that fit within a token limit.
    
    Strategy:
        1. First, try to split at paragraph boundaries (\n\n)
        2. If a paragraph is too long, split at sentence boundaries (.!?)
        3. If a sentence is still too long, hard-split by token count
    
    This preserves document structure as much as possible while guaranteeing
    each chunk fits within the model's context window.
    
    Args:
        text: The full document text to split
        tokenizer: HuggingFace tokenizer (e.g., from google/translategemma-12b-it)
        max_tokens: Maximum tokens per chunk. Default 1500 leaves room for
                    the prompt template (~150 tokens) within a 2K context window.
    
    Returns:
        List of text chunks, each guaranteed to be <= max_tokens
    """
    
    # =========================================================================
    # STEP 1: Normalize line endings and split into paragraphs
    # =========================================================================
    # Convert Windows (\r\n) and old Mac (\r) line endings to Unix (\n)
    # Then split on double newlines, which typically indicate paragraph breaks
    paragraphs = text.replace('\r\n', '\n').split('\n\n')
    
    # =========================================================================
    # STEP 2: Initialize accumulators
    # =========================================================================
    chunks = []           # Final list of text chunks to return
    current_chunk = []    # List of paragraphs in the chunk we're building
    current_tokens = 0    # Running token count for current_chunk
    
    # =========================================================================
    # STEP 3: Process each paragraph
    # =========================================================================
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue  # Skip empty paragraphs
        
        # Count how many tokens this paragraph contains
        para_tokens = count_tokens(para, tokenizer)
        
        # ---------------------------------------------------------------------
        # CASE A: Paragraph itself exceeds the token limit
        # We need to split it into smaller pieces
        # ---------------------------------------------------------------------
        if para_tokens > max_tokens:
            
            # First, save whatever we've accumulated so far
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split the paragraph into sentences using regex
            # (?<=[.!?]) is a lookbehind that matches position after .!?
            # \s+ matches one or more whitespace characters
            # This keeps the punctuation attached to the sentence
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            # Accumulators for building sentence-based chunks
            sent_chunk = []    # Sentences in current sub-chunk
            sent_tokens = 0    # Token count for sent_chunk
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                # Count tokens in this sentence
                st = count_tokens(sent, tokenizer)
                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # CASE A1: Single sentence exceeds limit (e.g., run-on sentence)
                # Last resort: split by raw token count
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                if st > max_tokens:
                    # Save accumulated sentences first
                    if sent_chunk:
                        chunks.append(' '.join(sent_chunk))
                        sent_chunk = []
                        sent_tokens = 0
                    
                    # Tokenize the sentence into token IDs
                    tokens = tokenizer.encode(sent, add_special_tokens=False)
                    
                    # Split token list into max_tokens-sized pieces
                    for i in range(0, len(tokens), max_tokens):
                        chunk_tokens = tokens[i:i + max_tokens]
                        # Decode back to text
                        # Note: This may split mid-word, which isn't ideal but
                        # guarantees we stay within the token limit
                        chunks.append(tokenizer.decode(chunk_tokens))
                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # CASE A2: Adding this sentence would exceed limit
                # Save current sentence chunk and start a new one
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                elif sent_tokens + st > max_tokens:
                    chunks.append(' '.join(sent_chunk))
                    sent_chunk = [sent]
                    sent_tokens = st
                
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # CASE A3: Sentence fits, add it to the current sentence chunk
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                else:
                    sent_chunk.append(sent)
                    sent_tokens += st
            
            # Don't forget any remaining sentences
            if sent_chunk:
                chunks.append(' '.join(sent_chunk))
        
        # ---------------------------------------------------------------------
        # CASE B: Adding this paragraph would exceed the limit
        # Save current chunk and start a new one with this paragraph
        # ---------------------------------------------------------------------
        elif current_tokens + para_tokens > max_tokens:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        
        # ---------------------------------------------------------------------
        # CASE C: Paragraph fits in current chunk
        # Just add it to the accumulator
        # ---------------------------------------------------------------------
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # =========================================================================
    # STEP 4: Don't forget the last chunk!
    # =========================================================================
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def count_tokens(text: str, tokenizer) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The string to tokenize
        tokenizer: A HuggingFace tokenizer instance
    
    Returns:
        Number of tokens (int)
    
    Note:
        add_special_tokens=False excludes BOS/EOS tokens from the count,
        giving us just the content tokens.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))

def chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """
    Split text into chunks at paragraph boundaries.
    Aims for chunks under max_chars while keeping paragraphs intact.
    
    Args:
        text: The text to split
        max_chars: Maximum characters per chunk (default 4000, ~1000 tokens)
    
    Returns:
        List of text chunks
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_len = len(para)
        
        # If single paragraph exceeds limit, split by sentences
        if para_len > max_chars:
            # First, save current chunk if any
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long paragraph by sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = []
            sentence_chunk_len = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sent_len = len(sentence)
                
                if sent_len > max_chars:
                    # Save accumulated sentences
                    if sentence_chunk:
                        chunks.append(' '.join(sentence_chunk))
                        sentence_chunk = []
                        sentence_chunk_len = 0
                    
                    # Hard split very long sentence
                    for i in range(0, sent_len, max_chars):
                        chunks.append(sentence[i:i + max_chars])
                
                elif sentence_chunk_len + sent_len + 1 > max_chars:
                    # Save and start new sentence chunk
                    chunks.append(' '.join(sentence_chunk))
                    sentence_chunk = [sentence]
                    sentence_chunk_len = sent_len
                else:
                    sentence_chunk.append(sentence)
                    sentence_chunk_len += sent_len + 1
            
            # Don't forget remaining sentences
            if sentence_chunk:
                chunks.append(' '.join(sentence_chunk))
        
        elif current_length + para_len + 2 > max_chars:
            # Current chunk is full, start new one
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_length = para_len
        else:
            # Add to current chunk
            current_chunk.append(para)
            current_length += para_len + 2
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def build_translation_prompt(text: str, source_lang: str, target_lang: str) -> str:
    """Build a raw text translation prompt (used by Ollama, optionally by HF)."""
    source_name = get_language_name(source_lang)
    target_name = get_language_name(target_lang)
    return f"""You are a professional {source_name} ({source_lang}) to {target_name} ({target_lang}) translator. Your goal is to accurately convey the meaning and nuances of the original {source_name} text while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.
Produce only the {target_name} translation, without any additional explanations or commentary. Please translate the following {source_name} text into {target_name}:


{text}"""


class TranslateGemma(ABC):
    """Base class for TranslateGemma backends."""

    def __init__(self, tokenizer, max_chunk_tokens: int = MAX_CHUNK_TOKENS):
        self.tokenizer = tokenizer
        self.max_chunk_tokens = max_chunk_tokens

    def translate(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate text, chunking if necessary for long documents."""

        # If text is short enough, translate directly
        if count_tokens(text, self.tokenizer) <= self.max_chunk_tokens:
            return self._translate_chunk(text, source_lang, target_lang)

        # Otherwise, chunk and translate
        chunks = chunk_text_by_tokens(text, self.tokenizer, max_tokens=self.max_chunk_tokens)
        print(f"    Document is long — splitting into {len(chunks)} chunks...")

        translated_chunks = []
        for i, chunk in enumerate(chunks, 1):
            tokens = count_tokens(chunk, self.tokenizer)
            print(f"    Translating chunk {i}/{len(chunks)} ({tokens} tokens)...")
            translated = self._translate_chunk(chunk, source_lang, target_lang)
            translated_chunks.append(translated)

        return '\n\n'.join(translated_chunks)

    @abstractmethod
    def _translate_chunk(self, text: str, source_lang: str, target_lang: str) -> str:
        ...


class TranslateGemmaOllama(TranslateGemma):
    """TranslateGemma using Ollama backend."""

    def __init__(self, model_name: str = "translategemma:12b", tokenizer=None, max_chunk_tokens: int = MAX_CHUNK_TOKENS):
        super().__init__(tokenizer, max_chunk_tokens)
        self.model_name = model_name

        try:
            import ollama
            self.client = ollama
            # Test connection
            try:
                self.client.list()
            except Exception as e:
                print(f"Error: Cannot connect to Ollama. Make sure 'ollama serve' is running.")
                print(f"Details: {e}")
                sys.exit(1)
        except ImportError:
            print("Error: ollama package not installed.")
            print("Install with: uv add ollama")
            sys.exit(1)

    def _translate_chunk(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single chunk of text."""
        prompt = build_translation_prompt(text, source_lang, target_lang)

        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": 2048,  # max context window
                "temperature": 0.1,   # Low temperature for consistent translation
            }
        )

        return response["message"]["content"]


class TranslateGemmaHF(TranslateGemma):
    """TranslateGemma using Hugging Face Transformers backend."""

    def __init__(self, model_name: str = "google/translategemma-12b-it", tokenizer=None, max_chunk_tokens: int = MAX_CHUNK_TOKENS, batch_size: int = 1, use_prompt: bool = False):
        super().__init__(tokenizer, max_chunk_tokens)
        self.batch_size = batch_size
        self.use_prompt = use_prompt

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
        if self.use_prompt:
            return [{
                "role": "user",
                "content": build_translation_prompt(text, source_lang, target_lang),
            }]
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": source_lang,
                "target_lang_code": target_lang,
                "text": text,
            }],
        }]

    def _is_truncated(self, text: str) -> bool:
        """Check if output was likely truncated by hitting max_new_tokens."""
        return count_tokens(text, self.tokenizer) >= self.max_chunk_tokens

    def _retry_with_split(self, text: str, source_lang: str, target_lang: str) -> str:
        """Re-translate a chunk by splitting it approximately in half."""
        input_tokens = count_tokens(text, self.tokenizer)
        # Use 60% of input tokens (not 50%) to avoid over-splitting while
        # still leaving enough headroom to avoid re-truncation
        half = max(int(input_tokens * 0.6), 1)
        print(f"      Output appears truncated (hit {self.max_chunk_tokens} token limit), "
              f"splitting chunk ({input_tokens} tokens) and retrying...")
        sub_chunks = chunk_text_by_tokens(text, self.tokenizer, max_tokens=half)
        parts = []
        for j, sub in enumerate(sub_chunks, 1):
            print(f"      Sub-chunk {j}/{len(sub_chunks)} ({count_tokens(sub, self.tokenizer)} tokens)...")
            parts.append(self._translate_chunk(sub, source_lang, target_lang))
        return '\n\n'.join(parts)

    def _translate_chunk(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single chunk, retrying with smaller chunks if truncated."""
        messages = self._build_messages(text, source_lang, target_lang)
        output = self.pipe(text=messages, do_sample=False, pad_token_id=1, max_new_tokens=self.max_chunk_tokens)
        result = output[0]["generated_text"][-1]["content"]
        if self._is_truncated(result):
            return self._retry_with_split(text, source_lang, target_lang)
        return result

    def translate(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate text, using batched inference for multi-chunk documents."""
        if count_tokens(text, self.tokenizer) <= self.max_chunk_tokens:
            chunks = [text]
        else:
            chunks = chunk_text_by_tokens(text, self.tokenizer, max_tokens=self.max_chunk_tokens)
            print(f"    Document is long — splitting into {len(chunks)} chunks...")

        translated_chunks = []
        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start:batch_start + self.batch_size]
            batch_end = batch_start + len(batch)
            if len(chunks) > 1:
                print(f"    Translating chunks {batch_start + 1}-{batch_end}/{len(chunks)}...")
            batch_messages = [self._build_messages(c, source_lang, target_lang) for c in batch]
            outputs = self.pipe(text=batch_messages, do_sample=False, batch_size=len(batch), pad_token_id=1, max_new_tokens=self.max_chunk_tokens)
            for i, output in enumerate(outputs):
                result = output[0]["generated_text"][-1]["content"]
                if self._is_truncated(result):
                    result = self._retry_with_split(batch[i], source_lang, target_lang)
                translated_chunks.append(result)

        return '\n\n'.join(translated_chunks)


def translate_file(
    translator,
    input_path: Path,
    output_dir: Path,
    source_lang: Optional[str] = None,
    target_lang: str = "en",
    suffix: str = "_translated_{target_lang}",
) -> Optional[bool]:
    """Translate a single file. Returns True on success, False on failure, None if skipped."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    
    # Read the file
    content = None
    for encoding in ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "iso-8859-15", "latin-1"]:
        try:
            with open(input_path, "r", encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print(f"  Error: Could not decode file {input_path}")
        return False
    
    if not content.strip():
        print(f"  Skipping empty file")
        return None
    
    print(f"  File size: {len(content):,} characters")
    
    # Detect language if not provided
    detected_lang = source_lang
    if detected_lang is None:
        # Use a sample from middle of document for detection (more representative)
        sample_start = max(0, len(content) // 4)
        sample_end = min(len(content), sample_start + 1000)
        sample = content[sample_start:sample_end]
        
        detected_lang = detect_language(sample)
        if detected_lang is None:
            # Fallback: try beginning of document
            detected_lang = detect_language(content[:1000])
        
        if detected_lang is None:
            print(f"  Error: Could not detect language (text may be too short or mixed)")
            print(f"  Tip: Use --source-lang to specify the language manually")
            return False
        
        print(f"  Detected language: {get_language_name(detected_lang)} ({detected_lang})")
    else:
        print(f"  Source language: {get_language_name(detected_lang)} ({detected_lang})")
    
    # Skip if already in target language
    if detected_lang == target_lang:
        print(f"  Skipping: already in {get_language_name(target_lang)}")
        return None
    
    # Check if language is in our known list
    if detected_lang not in LANGUAGE_NAMES:
        print(f"  Note: '{detected_lang}' not in common language list, but attempting anyway...")
    
    # Translate
    print(f"  Translating to: {get_language_name(target_lang)} ({target_lang})...")
    try:
        translated = translator.translate(content, detected_lang, target_lang)
    except Exception as e:
        print(f"  Error during translation: {e}")
        traceback.print_exc()
        return False
    
    # Validate output
    if not translated or not translated.strip():
        print(f"  Error: Translation returned empty result")
        return False
    
    # Check if translation actually happened (rough heuristic)
    # If output is >90% similar to input, it probably wasn't translated
    if len(translated) > 100 and detected_lang != "en":
        # Simple check: if first 100 chars are identical, warn
        if content[:100] == translated[:100]:
            print(f"  Warning: Output appears identical to input - translation may have failed")
    
    # Save output
    output_path = output_dir / f"{input_path.stem}{suffix.format(target_lang=target_lang)}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
    
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Output size: {len(translated):,} characters")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Translate documents to English using TranslateGemma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Ollama (recommended, make sure 'ollama serve' is running):
  uv run python translate_documents.py ./documents
  
  # Using Hugging Face Transformers:
  uv run python translate_documents.py ./documents --backend huggingface
  
  # Specify source language (skip auto-detection):
  uv run python translate_documents.py ./documents --source-lang de
  
  # Use smaller model (for machines with less RAM):
  uv run python translate_documents.py ./documents --model translategemma:4b
  
  # Adjust chunk size (in tokens) for very long documents:
  uv run python translate_documents.py ./documents --chunk-size 1000
  
  # Translate to a different target language:
  uv run python translate_documents.py ./documents --target-lang fr
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing .txt files to translate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: input_dir/translated)"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "huggingface"],
        default="ollama",
        help="Backend to use (default: ollama)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: translategemma:12b for Ollama, google/translategemma-12b-it for HF)"
    )
    parser.add_argument(
        "--source-lang",
        type=str,
        default=None,
        help="Source language code (e.g., 'de' for German). If not provided, will auto-detect."
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="en",
        help="Target language code (default: en for English)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=MAX_CHUNK_TOKENS,
        help=f"Maximum tokens per chunk for long documents (default: {MAX_CHUNK_TOKENS}, max: {MAX_CHUNK_TOKENS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of chunks to translate in parallel (HF backend only, default: 1)"
    )
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        default=False,
        help="Use raw text prompt instead of chat template (HF backend only, for comparison)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_translated_{target_lang}",
        help="Suffix for output filenames before .txt (default: '_translated_{target_lang}'). "
             "Use {target_lang} as a placeholder for the target language code."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded HuggingFace models (default: HF_HOME env var)"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        default=False,
        help="Allow downloading models from HuggingFace Hub (default: offline mode, models must already be cached)"
    )

    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} does not exist")
        sys.exit(1)
    
    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)

    if args.chunk_size > MAX_CHUNK_TOKENS:
        print(f"Warning: --chunk-size {args.chunk_size} exceeds recommended maximum of {MAX_CHUNK_TOKENS} tokens, clamping")
        args.chunk_size = MAX_CHUNK_TOKENS

    # Set output directory
    output_dir = args.output_dir or (args.input_dir / "translated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize translator
    print("=" * 60)
    print("TranslateGemma Document Translator")
    print("=" * 60)
    print(f"\nBackend: {args.backend}")
    print(f"Chunk size: {args.chunk_size} tokens")

    if args.backend == "ollama":
        model_name = args.model or "translategemma:12b"
        # Map Ollama model names to HF model IDs for the tokenizer
        OLLAMA_TO_HF = {
            "translategemma:4b": "google/translategemma-4b-it",
            "translategemma:12b": "google/translategemma-12b-it",
            "translategemma:27b": "google/translategemma-27b-it",
        }
        tokenizer_model = OLLAMA_TO_HF.get(model_name)
        if tokenizer_model is None:
            print(f"Error: Unknown Ollama model '{model_name}'. "
                  f"Known models: {', '.join(OLLAMA_TO_HF.keys())}")
            sys.exit(1)
        tokenizer = load_tokenizer(tokenizer_model, fetch=args.fetch)
        print(f"Model: {model_name}")
        print("\nInitializing Ollama backend...")
        translator = TranslateGemmaOllama(model_name, tokenizer=tokenizer, max_chunk_tokens=args.chunk_size)
    else:
        model_name = args.model or "google/translategemma-12b-it"
        tokenizer = load_tokenizer(model_name, fetch=args.fetch)
        print(f"Model: {model_name}")
        print("\nInitializing Hugging Face backend...")
        translator = TranslateGemmaHF(model_name, tokenizer=tokenizer, max_chunk_tokens=args.chunk_size, batch_size=args.batch_size, use_prompt=args.use_prompt)

    # Find all .txt files
    txt_files = sorted(args.input_dir.glob("*.txt"))
    if not txt_files:
        print(f"\nNo .txt files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(txt_files)} .txt file(s) to process")
    print(f"Output directory: {output_dir}")
    
    # Process each file
    success_count = 0
    skipped_count = 0
    failed_files = []

    for txt_file in txt_files:
        result = translate_file(
            translator,
            txt_file,
            output_dir,
            args.source_lang,
            args.target_lang,
            args.suffix,
        )
        if result is True:
            success_count += 1
        elif result is None:
            skipped_count += 1
        else:
            failed_files.append(txt_file.name)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Translation complete!")
    print(f"Successfully translated: {success_count}/{len(txt_files)} files")
    if skipped_count:
        print(f"Skipped: {skipped_count} file(s)")

    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
