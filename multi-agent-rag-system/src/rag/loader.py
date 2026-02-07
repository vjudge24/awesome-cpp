"""Document loaders for various file formats.

Supports PDF, Markdown, plain text, and URL content ingestion.
Each loader normalizes content into a standard (text, metadata) tuple.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LoadedDocument:
    """Normalized document content with metadata."""

    text: str
    metadata: dict[str, str]


class TextLoader:
    """Load plain text or Markdown files."""

    SUPPORTED = {".txt", ".md", ".markdown", ".rst"}

    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        if p.suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        text = p.read_text(encoding="utf-8")
        return LoadedDocument(
            text=text,
            metadata={"source": p.name, "format": p.suffix.lstrip(".")},
        )


class PDFLoader:
    """Load PDF files using pdfplumber (if available) or PyPDF2 fallback.

    Extracts text page-by-page with page number metadata.
    """

    def load(self, path: str | Path) -> LoadedDocument:
        p = Path(path)
        try:
            return self._load_with_pdfplumber(p)
        except ImportError:
            return self._load_with_pypdf(p)

    def _load_with_pdfplumber(self, path: Path) -> LoadedDocument:
        import pdfplumber

        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[Page {i + 1}]\n{text}")

        return LoadedDocument(
            text="\n\n".join(pages),
            metadata={
                "source": path.name,
                "format": "pdf",
                "total_pages": str(len(pages)),
            },
        )

    def _load_with_pypdf(self, path: Path) -> LoadedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text}")

        return LoadedDocument(
            text="\n\n".join(pages),
            metadata={
                "source": path.name,
                "format": "pdf",
                "total_pages": str(len(pages)),
            },
        )


class URLLoader:
    """Load content from a URL, stripping HTML tags for clean text.

    Uses httpx for async-compatible HTTP requests.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def load(self, url: str) -> LoadedDocument:
        logger.info("url_loading", url=url)
        response = httpx.get(url, timeout=self._timeout, follow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" in content_type:
            text = self._strip_html(response.text)
        else:
            text = response.text

        return LoadedDocument(
            text=text,
            metadata={
                "source": url,
                "format": "url",
                "content_type": content_type,
                "status_code": str(response.status_code),
            },
        )

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        # Remove script and style blocks
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
        # Remove tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


def auto_load(source: str) -> LoadedDocument:
    """Auto-detect source type and load accordingly.

    Args:
        source: A file path or URL string.

    Returns:
        LoadedDocument with extracted text and metadata.
    """
    if source.startswith(("http://", "https://")):
        return URLLoader().load(source)

    path = Path(source)
    if path.suffix == ".pdf":
        return PDFLoader().load(path)
    elif path.suffix in TextLoader.SUPPORTED:
        return TextLoader().load(path)
    else:
        # Fallback to text loader
        return LoadedDocument(
            text=path.read_text(encoding="utf-8"),
            metadata={"source": path.name, "format": "unknown"},
        )
