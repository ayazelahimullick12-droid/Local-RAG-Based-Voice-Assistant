"""Splits streamed LLM token deltas into TTS-ready phrases as early as
possible, so playback can start well before the full reply is generated.
"""
import re
from typing import List

_SPLIT_CHARS = "।?!,;\n"
_SPLIT_RE = re.compile(f"([{re.escape(_SPLIT_CHARS)}])")


class SentenceChunker:
    """Buffers token deltas and yields phrase-sized chunks on punctuation.

    Chunks shorter than `min_chars` are held and merged with the next piece
    so TTS never receives a fragment too small to sound natural (spec: ~15
    chars minimum, first chunk sent as early as possible).
    """

    def __init__(self, min_chars: int = 15):
        self.min_chars = min_chars
        self._buffer = ""

    def push(self, delta: str) -> List[str]:
        self._buffer += delta
        parts = _SPLIT_RE.split(self._buffer)  # [text, delim, text, delim, ..., text]
        chunks: List[str] = []
        pending = ""
        idx = 0
        while idx + 1 < len(parts):
            pending += parts[idx] + parts[idx + 1]
            if len(pending.strip()) >= self.min_chars:
                chunks.append(pending.strip())
                pending = ""
            idx += 2
        self._buffer = pending + parts[idx] if idx < len(parts) else pending
        return chunks

    def flush(self) -> List[str]:
        tail = self._buffer.strip()
        self._buffer = ""
        return [tail] if tail else []
