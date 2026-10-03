from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class STTResult:
    text: str
    is_final: bool
    latency_ms: float = 0.0


class STTEngine(ABC):
    """Common interface for all STT backends so the pipeline, the streaming
    wrapper, and the benchmark script can swap engines via one config value.
    """

    model_id: str
    device: str

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        """audio: float32 mono PCM in [-1, 1]."""
        ...
