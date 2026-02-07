"""Tests for document loaders."""

import tempfile
from pathlib import Path

import pytest

from src.rag.loader import LoadedDocument, TextLoader, URLLoader, auto_load


class TestTextLoader:
    def test_load_txt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, this is a test document.")
            path = f.name

        loader = TextLoader()
        doc = loader.load(path)

        assert doc.text == "Hello, this is a test document."
        assert doc.metadata["format"] == "txt"
        Path(path).unlink()

    def test_load_markdown_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nParagraph text here.")
            path = f.name

        loader = TextLoader()
        doc = loader.load(path)

        assert "# Title" in doc.text
        assert doc.metadata["format"] == "md"
        Path(path).unlink()

    def test_unsupported_format_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name

        loader = TextLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(path)
        Path(path).unlink()

    def test_load_japanese_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("これはテストドキュメントです。日本語のコンテンツを含みます。")
            path = f.name

        loader = TextLoader()
        doc = loader.load(path)
        assert "日本語" in doc.text
        Path(path).unlink()


class TestURLLoader:
    def test_strip_html(self):
        loader = URLLoader()
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        result = loader._strip_html(html)
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_strip_html_removes_scripts(self):
        loader = URLLoader()
        html = "<html><script>alert('xss')</script><p>Safe content</p></html>"
        result = loader._strip_html(html)
        assert "alert" not in result
        assert "Safe content" in result


class TestAutoLoad:
    def test_auto_load_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Auto loaded text")
            path = f.name

        doc = auto_load(path)
        assert doc.text == "Auto loaded text"
        Path(path).unlink()

    def test_auto_load_markdown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Auto loaded")
            path = f.name

        doc = auto_load(path)
        assert "# Auto loaded" in doc.text
        Path(path).unlink()


class TestLoadedDocument:
    def test_creation(self):
        doc = LoadedDocument(text="content", metadata={"source": "test"})
        assert doc.text == "content"
        assert doc.metadata["source"] == "test"
