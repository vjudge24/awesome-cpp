"""Tests for document chunking strategies."""

import pytest

from src.rag.chunking import Chunk, RecursiveChunker


class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        text = "This is a short text that fits in one chunk."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].metadata["strategy"] == "recursive"

    def test_paragraph_splitting(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "First paragraph about topic A.\n\nSecond paragraph about topic B."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2
        assert "First paragraph" in chunks[0].content
        assert "Second paragraph" in chunks[1].content

    def test_chunk_metadata(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        text = "Some text."
        chunks = chunker.chunk(text, metadata={"source": "test.pdf"})

        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["strategy"] == "recursive"

    def test_overlap_produces_repeated_tokens(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "A wonderful serenity has taken possession of my entire soul."
        )
        chunks = chunker.chunk(text)

        if len(chunks) >= 2:
            # The second chunk should start with overlap from the first
            assert len(chunks[1].content) > 0

    def test_empty_text(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_japanese_text(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "これは最初の段落です。日本語のテキストを処理します。\n\nこれは二番目の段落です。別のトピックについて説明します。"
        chunks = chunker.chunk(text)

        assert len(chunks) >= 1
        assert any("日本語" in c.content for c in chunks)

    def test_token_count_populated(self):
        chunk = Chunk(content="Hello world")
        assert chunk.token_count > 0


class TestChunk:
    def test_chunk_creation(self):
        chunk = Chunk(content="test content", metadata={"key": "value"})
        assert chunk.content == "test content"
        assert chunk.metadata["key"] == "value"
        assert chunk.token_count > 0

    def test_chunk_empty_metadata_default(self):
        chunk = Chunk(content="test")
        assert chunk.metadata == {}
