"""Fakes incremental STT on top of a non-streaming engine: re-runs
transcription on the growing utterance buffer every `interval_ms` so a live
partial transcript is available while the user is still talking, and only a
short final pass is needed once VAD detects end-of-speech.
"""
import asyncio
from typing import Callable, Optional

import numpy as np

from .base import STTEngine, STTResult


class StreamingTranscriber:
    def __init__(self, engine: STTEngine, interval_ms: int = 300,
                 sample_rate: int = 16000,
                 on_partial: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.interval_s = interval_ms / 1000
        self.sample_rate = sample_rate
        self.on_partial = on_partial
        self._buffer = np.zeros(0, dtype=np.float32)
        self._task: Optional[asyncio.Task] = None
        self._busy = False
        self.latest_partial = ""

    def append(self, frame_int16: np.ndarray) -> None:
        self._buffer = np.concatenate(
            [self._buffer, frame_int16.astype(np.float32) / 32768.0]
        )

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self.latest_partial = ""

    def start(self) -> None:
        self.reset()
        self._task = asyncio.create_task(self._loop())

    async def stop_and_finalize(self) -> STTResult:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        result = await self.engine.transcribe(self._buffer, self.sample_rate)
        result.is_final = True
        return result

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                # skip if a call is already in flight, or too little audio yet
                if self._busy or len(self._buffer) < self.sample_rate * 0.3:
                    continue
                self._busy = True
                try:
                    result = await self.engine.transcribe(self._buffer.copy(), self.sample_rate)
                    self.latest_partial = result.text
                    if self.on_partial:
                        self.on_partial(result.text)
                finally:
                    self._busy = False
        except asyncio.CancelledError:
            pass
