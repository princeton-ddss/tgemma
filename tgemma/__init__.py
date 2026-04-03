"""
TranslateGemma Document Translation Package
============================================

Translates .txt files from various languages to English using Google's TranslateGemma model.
Supports automatic language detection and chunking for small context window.

Usage:
    tgemma ./my_documents
    tgemma chunk ./my_documents
"""

import os
import sys
from pathlib import Path

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

# Suppress noisy transformers log messages
import logging
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*pipelines sequentially on GPU.*")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")

# Public API (imports after os.environ setup is intentional)
from .chunking import MAX_CHUNK_TOKENS, chunk_text_by_tokens, count_tokens  # noqa: E402
from .detection import LANGUAGES, detect_language, get_language_name  # noqa: E402
from .extraction import SUPPORTED_EXTENSIONS, extract_text, get_supported_files  # noqa: E402
from .orchestration import (  # noqa: E402
    download_tokenizer,
    get_output_path,
    load_tokenizer,
    translate_file,
    translate_text,
)
from .translator import HuggingFaceTranslator, Translator  # noqa: E402
from .utils import SkippedFileError, TranslationError, read_file_with_fallback  # noqa: E402

__all__ = [
    # Chunking
    "MAX_CHUNK_TOKENS",
    "chunk_text_by_tokens",
    "count_tokens",
    # Detection
    "LANGUAGES",
    "detect_language",
    "get_language_name",
    # Extraction
    "SUPPORTED_EXTENSIONS",
    "extract_text",
    "get_supported_files",
    # Orchestration
    "load_tokenizer",
    "download_tokenizer",
    "translate_text",
    "translate_file",
    "get_output_path",
    # Translator
    "HuggingFaceTranslator",
    "Translator",
    # Utils
    "TranslationError",
    "SkippedFileError",
    "read_file_with_fallback",
]
