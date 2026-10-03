from abc import ABC, abstractmethod

import numpy as np


class TTSEngine(ABC):
    """Common interface for all TTS backends. None of these models stream
    internally, which is why sentence-level chunking happens upstream in
    the orchestrator — each call synthesizes one short phrase at a time.
    """

    sample_rate: int

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 PCM in [-1, 1] at self.sample_rate."""
        ...
