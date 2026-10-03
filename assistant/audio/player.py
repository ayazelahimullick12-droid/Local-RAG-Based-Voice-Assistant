"""Local speaker output via sounddevice/PortAudio.

Plays queued PCM chunks strictly in order on a single background task, which
is what naturally guarantees the "filler must not overlap the real answer"
requirement: whatever is enqueued next simply plays after whatever is
already playing. `clear()` interrupts immediately for barge-in.
"""
import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd

from .transport import AudioOutputTransport

_WRITE_BLOCK_FRAMES = 1024


class SpeakerOutput(AudioOutputTransport):
    def __init__(self, sample_rate: int = 16000, device: Optional[str] = None):
        self.sample_rate = sample_rate
        self.device = device
        self._queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue()
        self._stream: Optional[sd.OutputStream] = None
        self._drain_task: Optional[asyncio.Task] = None
        self._interrupt = asyncio.Event()
        self._playing = False

    async def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", device=self.device
        )
        self._stream.start()
        self._drain_task = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            self._drain_task = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def enqueue(self, pcm: np.ndarray) -> None:
        await self._queue.put(np.asarray(pcm, dtype=np.float32))

    async def clear(self) -> None:
        self._interrupt.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._stream is not None:
            # abort() flushes the hardware buffer and unblocks any in-flight
            # write() immediately; start() re-arms the stream for the next chunk.
            await asyncio.to_thread(self._stream.abort)
            await asyncio.to_thread(self._stream.start)
        self._playing = False

    def is_playing(self) -> bool:
        return self._playing

    async def _drain(self) -> None:
        while True:
            pcm = await self._queue.get()
            self._playing = True
            self._interrupt.clear()
            try:
                for i in range(0, len(pcm), _WRITE_BLOCK_FRAMES):
                    if self._interrupt.is_set() or self._stream is None:
                        break
                    block = pcm[i : i + _WRITE_BLOCK_FRAMES]
                    await asyncio.to_thread(self._stream.write, block)
            except Exception:
                pass  # e.g. PortAudioError from an abort() mid-write during barge-in
            finally:
                self._playing = False
