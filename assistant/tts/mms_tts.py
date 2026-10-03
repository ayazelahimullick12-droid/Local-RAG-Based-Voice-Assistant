"""facebook/mms-tts-ben — Meta MMS VITS checkpoint for Bengali.

CAUTION: license is CC-BY-NC-4.0 (non-commercial) — confirmed during
research, corrects an unstated assumption. Do not ship this in anything
commercial; it's a fallback/quality-comparison engine here.

Verified: 16000 Hz sample rate (from the model's own config.json), loads
through plain transformers VitsModel + AutoTokenizer (no special tokenizer
class needed). Fixed seed keeps the voice consistent across turns, since
VITS sampling is stochastic by default.
"""
import asyncio

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

from .base import TTSEngine


class MMSTTSEngine(TTSEngine):
    def __init__(self, model_id: str = "facebook/mms-tts-ben", device: str = "cuda",
                 seed: int = 42, **_ignored):
        self.model_id = model_id
        self.device = device
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.sample_rate = self.model.config.sampling_rate

    async def warmup(self) -> None:
        await self.synthesize("হ্যালো")

    def _synth_sync(self, text: str) -> np.ndarray:
        torch.manual_seed(self.seed)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        return output.squeeze().float().cpu().numpy()

    async def synthesize(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._synth_sync, text)
