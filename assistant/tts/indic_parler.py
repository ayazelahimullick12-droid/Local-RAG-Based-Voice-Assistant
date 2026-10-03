"""ai4bharat/indic-parler-tts (Apache-2.0, 0.9B params — corrected from an
initially-assumed 1.5B) — voice-conditioned TTS with named Bengali voices;
Arjun/Aditi confirmed as the recommended pair during research. More
expressive than the VITS engines but slower; per spec, only worth keeping
if quality clearly beats EMTIAZZ and time-to-first-audio stays under
~400ms — benchmark before defaulting to this.

Requires the `parler-tts` package (installed from GitHub, not PyPI — see
requirements-tts-parler.txt). Imported lazily so the app can run with a
different TTS_ENGINE without this optional, fragile dependency installed.
"""
import asyncio

import numpy as np
import torch
from transformers import AutoTokenizer

from .base import TTSEngine


class IndicParlerTTSEngine(TTSEngine):
    def __init__(self, model_id: str = "ai4bharat/indic-parler-tts", device: str = "cuda",
                 voice_description: str = (
                     "Aditi speaks in Bengali with a clear, warm, moderate-pace, neutral tone."
                 ), **_ignored):
        from parler_tts import ParlerTTSForConditionalGeneration  # lazy import, optional dep

        self.model_id = model_id
        self.device = device
        self.voice_description = voice_description

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self.sample_rate = self.model.config.sampling_rate

    async def warmup(self) -> None:
        await self.synthesize("হ্যালো")

    def _synth_sync(self, text: str) -> np.ndarray:
        desc_ids = self.description_tokenizer(
            self.voice_description, return_tensors="pt"
        ).input_ids.to(self.device)
        prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            generation = self.model.generate(input_ids=desc_ids, prompt_input_ids=prompt_ids)
        return generation.float().cpu().numpy().squeeze()

    async def synthesize(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._synth_sync, text)
