"""
Text extraction from various file formats.

Supports .txt, .pdf, and common image formats (.png, .jpg, .jpeg, .tiff, .bmp).
For PDFs, tries text extraction first and falls back to OCR if text is sparse.
"""

from pathlib import Path

from .utils import TranslationError, read_file_with_fallback

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
MIN_TEXT_THRESHOLD = 50  # Minimum characters to consider PDF text extraction successful

# Lazy-loaded EasyOCR reader (initialized on first use)
_ocr_reader = None


def get_supported_files(input_dir: Path) -> list[Path]:
    """Get all supported files from input directory."""
    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def extract_text(path: Path) -> str:
    """
    Extract text from a file based on its extension.

    Args:
        path: Path to the file.

    Returns:
        Extracted text as a single string.

    Raises:
        TranslationError: If extraction fails.
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _extract_txt(path)
    elif suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in IMAGE_EXTENSIONS:
        return _extract_image(path)
    else:
        raise TranslationError(f"Unsupported file type: {suffix}")


def _extract_txt(path: Path) -> str:
    """Extract text from a .txt file."""
    return read_file_with_fallback(path)


def _extract_pdf(path: Path) -> str:
    """
    Extract text from a PDF file.

    Strategy:
    1. Try pdfplumber for text-based PDFs
    2. If text is minimal (<50 chars), fall back to OCR
    3. Concatenate all pages with double newlines
    """
    import pdfplumber

    try:
        with pdfplumber.open(path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text.strip())

            combined = "\n\n".join(t for t in pages_text if t)

            # If text extraction yielded minimal results, try OCR
            if len(combined) < MIN_TEXT_THRESHOLD:
                print(f"  PDF text extraction yielded minimal text ({len(combined)} chars), trying OCR...")
                return _extract_pdf_ocr(path)

            return combined

    except Exception as e:
        # If pdfplumber fails, try OCR as fallback
        print(f"  PDF text extraction failed ({e}), trying OCR...")
        return _extract_pdf_ocr(path)


def _extract_pdf_ocr(path: Path) -> str:
    """Extract text from a PDF using OCR (for scanned documents)."""
    from pdf2image import convert_from_path

    try:
        images = convert_from_path(path)
        return _ocr_images(images)
    except Exception as e:
        raise TranslationError(f"Failed to extract text from PDF via OCR: {e}") from e


def _extract_image(path: Path) -> str:
    """Extract text from an image using OCR."""
    from PIL import Image

    try:
        img = Image.open(path)
        return _ocr_images([img])
    except Exception as e:
        raise TranslationError(f"Failed to extract text from image: {e}") from e


def _ocr_images(images: list) -> str:
    """
    Run OCR on a list of PIL images.

    Args:
        images: List of PIL Image objects.

    Returns:
        Concatenated text from all images, separated by double newlines.
    """
    global _ocr_reader

    if _ocr_reader is None:
        import easyocr

        # Initialize with common languages; EasyOCR will download models on first use
        # Using a broad set to match TranslateGemma's multilingual capabilities
        print("  Initializing OCR (first run may download language models)...")
        _ocr_reader = easyocr.Reader(
            ["en", "es", "fr", "de", "pt", "it", "pl", "nl", "cs", "da", "fi", "el", "hu", "sv", "tr", "ko"],
            gpu=True,
            verbose=False,
        )

    pages_text = []
    for i, img in enumerate(images, 1):
        # EasyOCR expects numpy array or file path
        import numpy as np

        img_array = np.array(img)
        results = _ocr_reader.readtext(img_array)

        # Extract text from results (each result is [bbox, text, confidence])
        text = " ".join(r[1] for r in results)
        pages_text.append(text.strip())

        if len(images) > 1:
            print(f"    OCR page {i}/{len(images)} complete")

    return "\n\n".join(t for t in pages_text if t)
