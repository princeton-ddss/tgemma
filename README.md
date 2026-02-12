# TranslateGemma Document Translation Tool

Batch-translate `.txt` documents into English (or another target language) using Google's [TranslateGemma models](https://blog.google/innovation-and-ai/technology/developers-tools/translategemma/) via Ollama or HuggingFace. Supports automatic language detection and token-aware chunking to work within the small, 2048-token context window.

## How it works

1. Each `.txt` file in an input directory is read (with automatic file encoding detection).
2. The source language is detected via `langdetect` (or supplied with `--source-lang`).
3. If the document exceeds the input token budget (900 tokens by default), it is split into chunks at paragraph and/or sentence boundaries using the TranslateGemma tokenizer.
4. Chunks are sent to the chosen backend (Ollama API or HF Transformers pipeline). The HF backend also detects possibly-truncated output (if output tokens >> input tokens) and automatically re-splits and retries.
5. Translated chunks are reassembled and written to the output directory as `<filename><suffix>.txt` (default: `<filename>_translated_en.txt`).

## Supported languages

Explicitly supports Spanish, German, French, Portuguese, Czech, Danish, Finnish, Greek, Hungarian, Hebrew, Italian, Norwegian, Polish, Slovak, Swedish, Turkish, Korean, Catalan, and Basque.

The tool will also support anything else `langdetect` can identify and which the model supports (on a best-effort basis -- see [TranslateGemma technical report](https://arxiv.org/pdf/2601.09012)).

## Project structure

```
translation-tool/
  translate_documents.py   # Main translation script
  chunk_documents.py       # (Optional) utility to split docs into chunks
  pyproject.toml           # Project metadata & dependencies list (`uv`)
  input/                   # 1,219 input .txt files (~135 MB)
  translate.slurm          # Recommended slurm job params (edit as needed)
  uv.lock                  
  .gitignore
```

## Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- HuggingFace account with token, accepting terms for (gated) `translategemma` model family

### Setup 

```bash
git clone <repo-url> && cd translation-tool
uv sync
```

### Ollama backend (default)

1. Install [Ollama](https://ollama.com)
2. Pull the translation model. For example:

   ```bash
   ollama pull translategemma:12b
   ```
3. Download the tokenizer from HuggingFace (needed for token-aware chunking). This is a gated model, so create a HuggingFace account (if necessary), accept the terms via a [card page](https://huggingface.co/google/translategemma-4b-it) for at least one of the models in this family, and then authenticate. For example:

   ```bash
   uv run hf auth login
   uv run hf download google/translategemma-12b-it --include "tokenizer*" "special_tokens_map.json"
   ```

   If caching to a custom directory (e.g. on a shared filesystem), set `HF_HOME=/path/to/cache` before `uv`.
   NOTE: Alternatively, the script will download the tokenizer automatically when run with `--fetch`, if a token is present in the `HF_HOME` directory.
4. Start the Ollama server in a separate terminal:

   ```bash
   ollama serve
   ```

### Hugging Face backend

1. Requires a GPU with enough VRAM to load the model (~24 GB for 12B in bfloat16).
2. Authenticate with HuggingFace (gated model; see above):

   ```bash
   uv run hf auth login
   ```
3. Download a full model (includes tokenizer). For example:

   ```bash
   uv run hf download google/translategemma-12b-it
   ```
   Or to a custom cache directory, as described above.

## Usage

### Translate documents (Ollama -- default)

```bash
# Make sure Ollama is serving first!
uv run python translate_documents.py ./input
```

### Translate documents (Hugging Face Transformers)

```bash
uv run python translate_documents.py ./input --backend huggingface
```

### Optional arguments

```
--backend {ollama,huggingface}   Backend to use (default: ollama)
--model MODEL                    Model name (default: translategemma:12b / google/translategemma-12b-it; alternatives: 4b, 27b)
--source-lang CODE               ISO 639-1 source language code (default: none, uses auto-detection)
--target-lang CODE               Target language code (default: en)
--chunk-size TOKENS              Max tokens per chunk (default: 900, recommended max: 900)
--batch-size N                   Chunks to translate in parallel (HF backend only, default: 1)
--use-prompt                     Use raw text prompt instead of chat template (HF backend only; not recommended; for testing)
--output-dir DIR                 Output directory (default: input_dir/translated)
--suffix SUFFIX                  Suffix for output filenames before .txt (default: _translated_{target_lang}).
                                 Use {target_lang} as a placeholder for the target language code.
--cache-dir DIR                  Directory to cache downloaded HuggingFace models
                                 (default: HF_HOME env var)
--fetch                          Allow downloading models from HuggingFace Hub
                                 (default: offline mode, tokenizer and/or model must already be cached)
```

## Useful document chunker (no translation)

Useful for inspecting how documents will be split before running a full translation, debugging, etc.

```bash
uv run python chunk_documents.py ./input
uv run python chunk_documents.py ./input --chunk-size 500 --output-dir ./chunks
```

### Optional arguments

```
--output-dir DIR                 Output directory (default: input_dir/chunks)
--target-lang CODE               Files detected as this language are skipped (default: en)
--chunk-size TOKENS              Max tokens per chunk (default: 900, recommended max: 900)
--model MODEL                    HF model to load the tokenizer from (default: google/translategemma-12b-it)
--cache-dir DIR                  Directory to cache downloaded HuggingFace model tokenizer (default: HF_HOME env var)
--fetch                          Allow downloading tokenizer from HuggingFace Hub
                                 (default: offline mode, tokenizer must already be cached)
```

## Cluster deployment

Cluster deployment assumes use of the HuggingFace backend. This project tries to isolate the HuggingFace installables and cache (tokens, tokenizers, and models) in the project from whatever defaults you may have on your system. We do this to be friendly, especially in a cluster environment (where the default `HF_HOME` is in `/home`, a bad idea because the model weights are large).

The recommended cluster workflow is as follows:

### Setup

On login or viz node, as needed, install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Navigate to project directory:

```bash
# or /scratch/network on Adroit
cd /scratch/gpfs/your-net-id/some-project-dir
```

Install package dependencies:

```bash
uv python install 3.13
uv sync
```

Log in and install HF objects:

```bash
HF_HOME=./.hf uv run hf auth login
HF_HOME=./.hf uv run hf download google/translategemma-27b-it
```

Now modify the slurm script as desired and submit the job using the same model as the one you just downloaded (or another you have already downloaded to `./.hf`):

```bash
sbatch translate.slurm
```

## CHANGELOG
