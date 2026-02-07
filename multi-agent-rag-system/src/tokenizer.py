"""Token counting utility with graceful fallback.

Tries tiktoken first; if unavailable (e.g., no network to download
the encoding file), falls back to a character-based approximation.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_encoder = None
_initialized = False


def _init_encoder() -> None:
    global _encoder, _initialized
    if _initialized:
        return
    _initialized = True
    try:
        import tiktoken

        _encoder = tiktoken.encoding_for_model("gpt-4o")
        logger.debug("tokenizer_loaded", backend="tiktoken")
    except Exception:
        _encoder = None
        logger.debug("tokenizer_fallback", backend="char_approx")


def count_tokens(text: str) -> int:
    """Count tokens in text. Uses tiktoken if available, else ~4 chars/token."""
    _init_encoder()
    if _encoder is not None:
        return len(_encoder.encode(text))
    # Approximation: ~4 characters per token for English, ~2 for CJK
    return max(1, len(text) // 4)


def encode(text: str) -> list[int]:
    """Encode text to token IDs. Falls back to per-character encoding."""
    _init_encoder()
    if _encoder is not None:
        return _encoder.encode(text)
    return [ord(c) for c in text]


def decode(tokens: list[int]) -> str:
    """Decode token IDs back to text."""
    _init_encoder()
    if _encoder is not None:
        return _encoder.decode(tokens)
    return "".join(chr(t) for t in tokens)
