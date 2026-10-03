"""Local USB microphone input via sounddevice/PortAudio."""
import asyncio
from typing import AsyncIterator, Optional

import numpy as np
import sounddevice as sd

from .transport import AudioInputTransport


class MicrophoneInput(AudioInputTransport):
    def __init__(self, sample_rate: int = 16000, frame_samples: int = 512,
                 device: Optional[str] = None):
        # frame_samples=512 @ 16kHz = 32ms, the chunk size Silero VAD expects.
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self._queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=200)
        self._stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _put(self, chunk: np.ndarray) -> None:
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass  # drop rather than build up latency if the consumer stalls

    def _callback(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._put, chunk)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            yield await self._queue.get()
