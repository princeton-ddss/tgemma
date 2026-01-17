#!/usr/bin/env python3
"""
TranslateGemma Document Translation Script
==========================================
Translates .txt files from various languages to English using Google's TranslateGemma model.
Supports automatic language detection.

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
from pathlib import Path
from typing import Optional

# Language detection
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent language detection
DetectorFactory.seed = 0

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
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
    "hi": "Hindi",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ca": "Catalan",
    "eu": "Basque",
    "gl": "Galician",
    "af": "Afrikaans",
    "sw": "Swahili",
    "is": "Icelandic",
    "mt": "Maltese",
    "cy": "Welsh",
    "ga": "Irish",
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


class TranslateGemmaOllama:
    """TranslateGemma using Ollama backend."""
    
    def __init__(self, model_name: str = "translategemma:12b"):
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
    
    def translate(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate text using TranslateGemma via Ollama."""
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)
        
        # TranslateGemma prompt format (note: two blank lines before text)
        prompt = f"""You are a professional {source_name} ({source_lang}) to {target_name} ({target_lang}) translator. Your goal is to accurately convey the meaning and nuances of the original {source_name} text while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.
Produce only the {target_name} translation, without any additional explanations or commentary. Please translate the following {source_name} text into {target_name}:


{text}"""
        
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response["message"]["content"]


class TranslateGemmaHF:
    """TranslateGemma using Hugging Face Transformers backend."""
    
    def __init__(self, model_name: str = "google/translategemma-12b-it"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        self.model_name = model_name
        self.torch = torch
        
        # Determine device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon (MPS) backend")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA backend")
        else:
            self.device = "cpu"
            print("Warning: Using CPU - this will be slow!")
        
        # Determine dtype
        if self.device == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16
        
        print(f"Loading model: {model_name}...")
        print("(This may take a few minutes on first run as the model downloads)")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=self.dtype
        )
        print("Model loaded!")
    
    def translate(self, text: str, source_lang: str, target_lang: str = "en") -> str:
        """Translate text using TranslateGemma via HF Transformers."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.dtype)
        
        input_len = len(inputs['input_ids'][0])
        
        with self.torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False
            )
        
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        return decoded


def translate_file(
    translator,
    input_path: Path,
    output_dir: Path,
    source_lang: Optional[str] = None,
    target_lang: str = "en"
) -> bool:
    """Translate a single file."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    
    # Read the file
    content = None
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]:
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
        return False
    
    # Detect language if not provided
    detected_lang = source_lang
    if detected_lang is None:
        detected_lang = detect_language(content)
        if detected_lang is None:
            print(f"  Error: Could not detect language (text may be too short)")
            print(f"  Tip: Use --source-lang to specify the language manually")
            return False
        print(f"  Detected language: {get_language_name(detected_lang)} ({detected_lang})")
    else:
        print(f"  Source language: {get_language_name(detected_lang)} ({detected_lang})")
    
    # Skip if already in target language
    if detected_lang == target_lang:
        print(f"  Skipping: already in {get_language_name(target_lang)}")
        return False
    
    # Check if language is supported
    if detected_lang not in LANGUAGE_NAMES:
        print(f"  Warning: '{detected_lang}' may not be in the primary 55 languages")
        print(f"  Attempting translation anyway...")
    
    # Translate
    print(f"  Translating to: {get_language_name(target_lang)} ({target_lang})...")
    try:
        translated = translator.translate(content, detected_lang, target_lang)
    except Exception as e:
        print(f"  Error during translation: {e}")
        return False
    
    # Save output
    output_path = output_dir / f"{input_path.stem}_translated_{target_lang}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translated)
    
    print(f"  ✓ Saved to: {output_path}")
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
  
  # Translate to a different target language:
  uv run python translate_documents.py ./documents --target-lang fr

Supported source languages (your list):
  Spanish (es), German (de), French (fr), Portuguese (pt), Czech (cs),
  Danish (da), Finnish (fi), Greek (el), Hungarian (hu), Hebrew (he),
  Italian (it), Norwegian (no), Polish (pl), Slovak (sk), Swedish (sv),
  Turkish (tr), Korean (ko)
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
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} does not exist")
        sys.exit(1)
    
    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir or (args.input_dir / "translated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize translator
    print("="*60)
    print("TranslateGemma Document Translator")
    print("="*60)
    print(f"\nBackend: {args.backend}")
    
    if args.backend == "ollama":
        model_name = args.model or "translategemma:12b"
        print(f"Model: {model_name}")
        print("\nInitializing Ollama backend...")
        translator = TranslateGemmaOllama(model_name)
    else:
        model_name = args.model or "google/translategemma-12b-it"
        print(f"Model: {model_name}")
        print("\nInitializing Hugging Face backend...")
        translator = TranslateGemmaHF(model_name)
    
    # Find all .txt files
    txt_files = sorted(args.input_dir.glob("*.txt"))
    if not txt_files:
        print(f"\nNo .txt files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(txt_files)} .txt file(s) to process")
    print(f"Output directory: {output_dir}")
    
    # Process each file
    success_count = 0
    for txt_file in txt_files:
        if translate_file(
            translator,
            txt_file,
            output_dir,
            args.source_lang,
            args.target_lang
        ):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Translation complete!")
    print(f"Successfully translated: {success_count}/{len(txt_files)} files")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
