"""Central typed config, loaded from config/.env. Every latency/model/VAD
knob the spec calls out lives here — nothing should be hardcoded elsewhere.
"""
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / "config" / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Audio -------------------------------------------------------------
    AUDIO_INPUT_DEVICE: Optional[str] = None
    AUDIO_OUTPUT_DEVICE: Optional[str] = None
    AUDIO_SAMPLE_RATE: int = 16000

    # VAD -----------------------------------------------------------------
    VAD_START_THRESHOLD: float = 0.5
    VAD_END_SILENCE_MS: int = 500
    VAD_MIN_SPEECH_MS: int = 200

    # STT -------------------------------------------------------------
    STT_ENGINE: Literal["bangla_asr", "nemo_conformer", "indic_conformer"] = "bangla_asr"
    STT_MODEL_ID: str = "bangla-speech-processing/BanglaASR"
    STT_DEVICE: str = "cuda"
    STT_DTYPE: str = "bfloat16"
    STT_LANGUAGE: str = "bn"
    STT_NEMO_FILENAME: Optional[str] = None
    STT_INDIC_DECODING: Literal["ctc", "rnnt"] = "ctc"
    STT_PARTIAL_INTERVAL_MS: int = 300

    # LLM -------------------------------------------------------------
    LLM_MODEL: str = "gemma4:26b"
    LLM_HOST: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.6
    LLM_NUM_PREDICT_NORMAL: int = 120
    LLM_NUM_PREDICT_GREETING: int = 40
    LLM_THINK: bool = False
    LLM_KEEP_ALIVE: str = "-1"

    # TTS -------------------------------------------------------------
    TTS_ENGINE: Literal["emtiazz_vits", "mms_tts", "indic_parler"] = "emtiazz_vits"
    TTS_DEVICE: str = "cuda"
    TTS_PARLER_VOICE_DESCRIPTION: str = (
        "Aditi speaks in Bengali with a clear, warm, moderate-pace, neutral tone."
    )

    # Filler phrases -----------------------------------------------------
    FILLER_ENABLED: bool = True
    FILLER_TRIGGER_MS: int = 700
    FILLER_PHRASES: str = (
        "একটু দেখি…,আচ্ছা বলছি…,হুম এক মিনিট…,ভালো প্রশ্ন…,"
        "ঠিক আছে দেখছি…,একটু সময় দিন…,বুঝেছি বলছি…,হ্যাঁ ভাবছি…"
    )

    # Barge-in -------------------------------------------------------------
    BARGE_IN_ENABLED: bool = True

    # RAG -------------------------------------------------------------
    RAG_ENABLED: bool = True
    RAG_KNOWLEDGE_DIR: str = "knowledge"
    RAG_CACHE_PATH: str = "knowledge/.embed_cache.json"
    RAG_CHUNK_CHARS: int = 900
    RAG_TOP_K: int = 3
    RAG_MIN_SCORE: float = 0.15
    RAG_EMBED_MODEL: str = "bge-m3"
    RAG_WEIGHT_EMBED: float = 0.65
    RAG_WEIGHT_TFIDF: float = 0.35
    RAG_WEIGHT_TFIDF_WORD: float = 0.6
    RAG_WEIGHT_TFIDF_CHAR: float = 0.4

    # Logging -------------------------------------------------------------
    LATENCY_LOG_PATH: str = "logs/latency.jsonl"
    LOG_LEVEL: str = "INFO"

    @property
    def filler_phrases_list(self) -> List[str]:
        return [p.strip() for p in self.FILLER_PHRASES.split(",") if p.strip()]
