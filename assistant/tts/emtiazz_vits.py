"""EMTIAZZ/bangladeshi-bangla-tts-vits — Bangladeshi-accented Bengali VITS
(Apache-2.0). PRIMARY TTS engine per the spec's Bengali-quality priority.

IMPORTANT correction from initial assumptions: this checkpoint is Coqui-VITS
format (raw pytorch_model.pth + config.json), NOT a huggingface transformers
checkpoint. It's loaded through Coqui's own `TTS` package
(`TTS.utils.synthesizer.Synthesizer`), confirmed against the training repo's
own inference.py (github.com/EMTIAZ-RUET/bangladeshi-bangla-tts-finetuning).
See requirements.txt for the `coqui-tts` package (community-maintained fork
of the now-largely-unmaintained coqui-ai/TTS).

The exact checkpoint/config filenames inside the HF repo were not confirmed
byte-for-byte during research (only the training repo's script was) — this
tries the conventional names first and falls back to scanning the repo file
listing.
"""
import asyncio

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

from .base import TTSEngine


def _resolve_file(model_id: str, filename: str, suffixes: tuple) -> str:
    try:
        return hf_hub_download(repo_id=model_id, filename=filename)
    except Exception:
        candidates = [f for f in list_repo_files(model_id) if f.endswith(suffixes)]
        if not candidates:
            raise FileNotFoundError(f"No {suffixes} file found in {model_id}")
        return hf_hub_download(repo_id=model_id, filename=candidates[0])


class EmtiazzVitsEngine(TTSEngine):
    def __init__(self, model_id: str = "EMTIAZZ/bangladeshi-bangla-tts-vits",
                 device: str = "cuda",
                 checkpoint_filename: str = "pytorch_model.pth",
                 config_filename: str = "config.json", **_ignored):
        from TTS.utils.synthesizer import Synthesizer  # coqui-tts

        checkpoint_path = _resolve_file(model_id, checkpoint_filename, (".pth", ".pth.tar"))
        config_path = _resolve_file(model_id, config_filename, ("config.json",))

        self.synthesizer = Synthesizer(
            tts_checkpoint=checkpoint_path,
            tts_config_path=config_path,
            use_cuda=(device == "cuda"),
        )
        self.sample_rate = self.synthesizer.output_sample_rate

    async def warmup(self) -> None:
        await self.synthesize("হ্যালো")

    def _synth_sync(self, text: str) -> np.ndarray:
        wav = self.synthesizer.tts(text)
        return np.asarray(wav, dtype=np.float32)

    async def synthesize(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._synth_sync, text)
