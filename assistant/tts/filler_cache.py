"""Pre-synthesizes short Bengali filler phrases at startup so a filler can
be played instantly (no TTS latency) if the real answer is slow to start.
"""
import random
from typing import List, Optional

import numpy as np

from .base import TTSEngine


class FillerCache:
    def __init__(self, engine: TTSEngine, phrases: List[str]):
        self.engine = engine
        self.phrases = phrases
        self._audio: List[np.ndarray] = []
        self._last_index: Optional[int] = None

    async def prewarm(self) -> None:
        self._audio = [await self.engine.synthesize(phrase) for phrase in self.phrases]

    def get_random(self) -> Optional[np.ndarray]:
        if not self._audio:
            return None
        choices = list(range(len(self._audio)))
        if self._last_index is not None and len(choices) > 1:
            choices.remove(self._last_index)
        idx = random.choice(choices)
        self._last_index = idx
        return self._audio[idx]
