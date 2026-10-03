"""ai4bharat/indic-conformer-600m-multilingual (MIT) — hybrid CTC/RNNT
Conformer, Bengali among 22 supported languages. Verified: does NOT need
NeMo — loads through plain transformers with trust_remote_code=True, and
the custom forward signature is `model(wav, "bn", "ctc"|"rnnt")`.

ai4bharat's custom wrapper code has changed shape across versions in the
past — if this raises a signature error, check the live model card's
example snippet and adjust `_transcribe_sync` accordingly.
"""
import asyncio
import time

import numpy as np
import torch
from transformers import AutoModel

from .base import STTEngine, STTResult


class IndicConformerEngine(STTEngine):
    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual",
                 device: str = "cuda", decoding: str = "ctc", **_ignored):
        self.model_id = model_id
        self.device = device
        self.decoding = decoding
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.model.eval()

    async def warmup(self) -> None:
        await self.transcribe(np.zeros(16000, dtype=np.float32), 16000)

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            text = self.model(wav, "bn", self.decoding)
        return text.strip() if isinstance(text, str) else str(text).strip()

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        start = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe_sync, audio, sample_rate)
        return STTResult(text=text, is_final=True, latency_ms=(time.perf_counter() - start) * 1000)
