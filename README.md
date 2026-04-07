# tgemma - TranslateGemma Document Translation

Batch-translate `.txt` documents into English (or another target language) using Google's [TranslateGemma models](https://huggingface.co/collections/google/translategemma-release-67c2e3ba5ae4c7007c8ae1a4) via HuggingFace Transformers. Supports automatic language detection and token-aware chunking to work within the 2048-token context window.

## How it works

1. Each `.txt` file in an input directory is read (with automatic encoding detection).
2. The source language is detected via `langdetect` (or supplied with `--source-lang`).
3. If the document exceeds the token budget (900 tokens by default), it is split into chunks at paragraph and/or sentence boundaries.
4. Chunks are translated via the HuggingFace Transformers pipeline. Truncated output is automatically detected and retried with smaller chunks.
5. Translated chunks are reassembled and written to the output directory.

## Supported languages

Spanish, German, French, Portuguese, Czech, Danish, Finnish, Greek, Hungarian, Hebrew, Italian, Norwegian, Polish, Slovak, Swedish, Turkish, Korean, Catalan, Basque, and others supported by `langdetect` and TranslateGemma (see [technical report](https://arxiv.org/pdf/2601.09012)).

## Project structure

```
tgemma/
  __init__.py       # Package exports
  chunking.py       # Token-aware text splitting
  cli.py            # Typer CLI
  detection.py      # Language detection
  orchestration.py  # Translation workflows
  translator.py     # HuggingFace translator
  utils.py          # File I/O, exceptions
```

## Prerequisites

- Python >= 3.13
- GPU with sufficient VRAM (~24 GB for 12B model in bfloat16)
- HuggingFace account with accepted terms for [TranslateGemma](https://huggingface.co/google/translategemma-12b-it)

## Running Locally

### Setup
```bash
git clone <repo-url> && cd tgemma
```

**With uv (recommended — no separate Python install needed):**
```bash
pip install uv        # skip if already installed
uv sync
```

**With conda:**
```bash
conda create -n tgemma python=3.13
conda activate tgemma
pip install -e .
```

Authenticate with HuggingFace (you'll be prompted to paste your token — get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) after accepting the [TranslateGemma license](https://huggingface.co/google/translategemma-12b-it)):
```bash
hf auth login
```

Download the model:
```bash
hf download google/translategemma-12b-it
```

### Usage

#### Translate documents

```bash
tgemma ./input
```

#### Options

```
tgemma --help

Options:
  --output-dir PATH      Output directory (default: input_dir/translated)
  --source-lang TEXT     Source language code (auto-detect if not provided)
  --target-lang TEXT     Target language code (default: en)
  --chunk-size INT       Maximum tokens per chunk (default: 900)
  --batch-size INT       Chunks to translate in parallel (default: 1)
  --model TEXT           HuggingFace model (default: google/translategemma-12b-it)
  --suffix TEXT          Output filename suffix (default: _translated_{target_lang})
  --fetch / --no-fetch   Allow downloading from HuggingFace Hub (default: no-fetch)
  --force / --no-force   Re-translate even if output exists (default: no-force)
```

#### Chunk documents (no translation)

Useful for inspecting how documents will be split:

```bash
tgemma chunk ./input
tgemma chunk ./input --chunk-size 500
```

## HPC Cluster Deployment

### Setup (login node)

**With conda:**
```bash
cd /scratch/gpfs/$USER/tgemma
conda create -n tgemma python=3.13
conda activate tgemma
pip install -e .
```

**With uv:**
```bash
cd /scratch/gpfs/$USER/tgemma
pip install uv        # skip if already installed
uv sync
```

Authenticate and download the model to local cache:
```bash
hf auth login
HF_HOME=./.hf hf download google/translategemma-27b-it
```

### Example Slurm script

```bash
#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

cd /scratch/gpfs/$USER/tgemma
conda activate tgemma
export HF_HOME=./.hf

tgemma --batch-size 25 --model google/translategemma-27b-it ./input
```

Or with uv (no activation needed):

```bash
cd /scratch/gpfs/$USER/tgemma
export HF_HOME=./.hf
uv run tgemma --output-dir ./output --batch-size 25 ./input
```

## Programmatic usage

```python
from tgemma import (
    HuggingFaceTranslator,
    load_tokenizer,
    translate_text,
    translate_file,
)

# Load tokenizer and translator
tokenizer = load_tokenizer("google/translategemma-12b-it")
translator = HuggingFaceTranslator(
    "google/translategemma-12b-it",
    tokenizer=tokenizer,
)

# Translate text
result = translate_text(
    "Hallo, wie geht es dir?",
    translator,
    source_lang="de",
    target_lang="en",
)
```
