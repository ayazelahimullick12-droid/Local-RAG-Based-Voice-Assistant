"""Abstract audio transport. Everything above this layer (VAD, STT, the
orchestrator) talks only to these two interfaces, never to sounddevice
directly — so a WebRTC transport (e.g. a LiveKit room) can be dropped in
later for remote use without touching pipeline code.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np


class AudioInputTransport(ABC):
    """Source of mono int16 PCM audio frames at a fixed sample rate."""

    sample_rate: int

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield fixed-size int16 mono numpy frames as they arrive."""
        ...


class AudioOutputTransport(ABC):
    """Sink for mono float32 PCM audio, queued and played in order."""

    sample_rate: int

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def enqueue(self, pcm: np.ndarray) -> None:
        """Queue float32 mono PCM in [-1, 1] for playback."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Drop everything queued/playing immediately — used for barge-in."""
        ...

    @abstractmethod
    def is_playing(self) -> bool: ...
