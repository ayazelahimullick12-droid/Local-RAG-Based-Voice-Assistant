"""kazalbrur/bangla-stt-conformer-120m-dialects — NeMo Conformer-CTC
(Apache-2.0 — corrected from an initially-assumed CC-BY-NC-4.0; 121.5M
params, strong on Bangladeshi regional dialects, ~27% WER, unpunctuated
output). Fallback/A-B STT, not the default.

Requires nemo_toolkit[asr] (see requirements-nemo.txt / docker/Dockerfile.nemo)
— imported lazily so the rest of the app runs fine without it installed.

Loading: `ASRModel.restore_from()` needs a local .nemo checkpoint file, not
a bare HF repo id, so this downloads one via huggingface_hub first. The
exact filename inside the HF repo wasn't confirmed during research; set
STT_NEMO_FILENAME explicitly if auto-detection below picks the wrong file.
"""
import asyncio
import os
import tempfile
import time
from typing import Optional

import numpy as np

from .base import STTEngine, STTResult


def _resolve_nemo_checkpoint(model_id: str, filename: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download, list_repo_files

    if filename:
        return hf_hub_download(repo_id=model_id, filename=filename)
    candidates = [f for f in list_repo_files(model_id) if f.endswith(".nemo")]
    if not candidates:
        raise FileNotFoundError(
            f"No .nemo file found in {model_id}; set STT_NEMO_FILENAME explicitly."
        )
    return hf_hub_download(repo_id=model_id, filename=candidates[0])


class NeMoConformerEngine(STTEngine):
    def __init__(self, model_id: str = "kazalbrur/bangla-stt-conformer-120m-dialects",
                 device: str = "cuda", nemo_filename: Optional[str] = None, **_ignored):
        self.model_id = model_id
        self.device = device

        import nemo.collections.asr as nemo_asr  # lazy import, heavy dependency

        checkpoint_path = _resolve_nemo_checkpoint(model_id, nemo_filename)
        self.model = nemo_asr.models.ASRModel.restore_from(
            checkpoint_path, map_location=device
        )
        self.model = self.model.to(device)
        self.model.eval()

    async def warmup(self) -> None:
        await self.transcribe(np.zeros(16000, dtype=np.float32), 16000)

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        import soundfile as sf

        # NeMo's transcribe() API takes file paths, not arrays — this is the
        # main reason this engine is not on the streaming hot path.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            path = f.name
        try:
            hyps = self.model.transcribe([path])
            hyp = hyps[0]
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
        finally:
            os.remove(path)
        return text.strip()

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        start = time.perf_counter()
        text = await asyncio.to_thread(self._transcribe_sync, audio, sample_rate)
        return STTResult(text=text, is_final=True, latency_ms=(time.perf_counter() - start) * 1000)
