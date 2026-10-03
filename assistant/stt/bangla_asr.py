"""bangla-speech-processing/BanglaASR — Whisper-small fine-tuned on Bengali
(MIT, ~4.58% WER on Common Voice — verified against the live model card).
Standard transformers Whisper loading path; the card's own example code
uses plain WhisperProcessor + WhisperForConditionalGeneration.generate(),
no special generation_config quirks.

Not natively streaming: assistant/stt/streaming.py re-runs this on the
growing utterance buffer every ~300ms to fake incremental transcription.
"""
import asyncio
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .base import STTEngine, STTResult


class BanglaASREngine(STTEngine):
    def __init__(self, model_id: str = "bangla-speech-processing/BanglaASR",
                 device: str = "cuda", dtype: str = "bfloat16", **_ignored):
        self.model_id = model_id
        self.device = device
        self.torch_dtype = getattr(torch, dtype)

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, attn_implementation="sdpa",
        ).to(device)
        self.model.eval()
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="bn", task="transcribe"
        )

    async def warmup(self) -> None:
        await self.transcribe(np.zeros(16000, dtype=np.float32), 16000)

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
        with torch.no_grad():
            ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                do_sample=False,
                num_beams=1,
                max_new_tokens=225,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        start = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe_sync, audio, sample_rate)
        return STTResult(text=text, is_final=True, latency_ms=(time.perf_counter() - start) * 1000)
