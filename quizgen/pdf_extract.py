"""Text extraction from PDF files (PyMuPDF with a pypdf fallback)."""
from __future__ import annotations

import io
from typing import List, Union

from .models import Page

PdfSource = Union[str, bytes]


def _extract_pymupdf(data: bytes) -> List[Page]:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # older PyMuPDF releases

    pages = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            pages.append(Page(number=i, text=page.get_text("text")))
    return pages


def _extract_pypdf(data: bytes) -> List[Page]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [Page(number=i, text=p.extract_text() or "") for i, p in enumerate(reader.pages, start=1)]


def extract_pages(source: PdfSource, page_range: tuple[int, int] | None = None) -> List[Page]:
    """Return the text of each page. `source` is a file path or raw PDF bytes.

    `page_range` is an inclusive, 1-indexed (start, end) filter.
    """
    data = source if isinstance(source, bytes) else open(source, "rb").read()
    try:
        pages = _extract_pymupdf(data)
    except ImportError:
        pages = _extract_pypdf(data)
    if page_range:
        start, end = page_range
        pages = [p for p in pages if start <= p.number <= end]
    if not any(p.text.strip() for p in pages):
        raise ValueError(
            "No extractable text found. The PDF is probably scanned; run OCR first "
            "(e.g. `ocrmypdf input.pdf output.pdf`)."
        )
    return pages
