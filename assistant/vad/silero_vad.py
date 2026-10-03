"""Silero VAD (CPU — it's tiny, no need for the GPU) with a small speech
start/end state machine on top, tuned by VAD_START_THRESHOLD /
VAD_END_SILENCE_MS / VAD_MIN_SPEECH_MS.

Loading: prefers the `silero-vad` pip package (verified current install
path — snakers4/silero-vad, MIT), falls back to torch.hub if the package
isn't installed.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import numpy as np
import torch


class VADEvent(Enum):
    NONE = auto()
    SPEECH_START = auto()
    SPEECH_END = auto()


@dataclass
class VADConfig:
    sample_rate: int = 16000
    start_threshold: float = 0.5
    end_silence_ms: int = 500
    min_speech_ms: int = 200
    frame_ms: int = 32  # matches 512 samples @ 16kHz, Silero's expected chunk size


def _load_silero_model():
    try:
        from silero_vad import load_silero_vad
        return load_silero_vad(onnx=False)
    except ImportError:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        return model


class SileroVAD:
    def __init__(self, config: VADConfig):
        self.config = config
        self.model = _load_silero_model()
        self.model.eval()
        self._speech_frames = 0
        self._silence_ms = 0
        self._in_speech = False

    def reset(self) -> None:
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()
        self._speech_frames = 0
        self._silence_ms = 0
        self._in_speech = False

    def frame_probability(self, frame_int16: np.ndarray) -> float:
        audio = torch.from_numpy(frame_int16.astype(np.float32) / 32768.0)
        with torch.no_grad():
            return float(self.model(audio, self.config.sample_rate).item())

    def process(self, frame_int16: np.ndarray) -> Tuple[VADEvent, float]:
        """Feed one fixed-size frame; returns (event, raw_speech_probability)."""
        prob = self.frame_probability(frame_int16)
        frame_ms = self.config.frame_ms
        event = VADEvent.NONE

        if prob >= self.config.start_threshold:
            self._speech_frames += 1
            self._silence_ms = 0
            if not self._in_speech and self._speech_frames * frame_ms >= self.config.min_speech_ms:
                self._in_speech = True
                event = VADEvent.SPEECH_START
        else:
            self._speech_frames = 0
            if self._in_speech:
                self._silence_ms += frame_ms
                if self._silence_ms >= self.config.end_silence_ms:
                    self._in_speech = False
                    event = VADEvent.SPEECH_END

        return event, prob
