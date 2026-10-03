"""Entry point: loads config, builds every engine (each with a warmup pass
so the first real turn isn't slow), prints a GPU memory table, then runs the
live mic/speaker loop. Run with: python -m assistant.main
"""
import asyncio
import logging

from .audio.mic_transport import MicrophoneInput
from .audio.player import SpeakerOutput
from .config import Settings
from .llm.client import OllamaLLMClient
from .llm.router import FastRouter
from .pipeline.latency import LatencyLogger
from .pipeline.orchestrator import VoiceAssistantOrchestrator
from .rag.embeddings import EmbeddingCache
from .rag.retrieval import HybridRetriever
from .stt.bangla_asr import BanglaASREngine
from .stt.indic_conformer import IndicConformerEngine
from .stt.nemo_conformer import NeMoConformerEngine
from .tts.emtiazz_vits import EmtiazzVitsEngine
from .tts.filler_cache import FillerCache
from .tts.indic_parler import IndicParlerTTSEngine
from .tts.mms_tts import MMSTTSEngine
from .vad.silero_vad import SileroVAD, VADConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("main")

STT_ENGINES = {
    "bangla_asr": BanglaASREngine,
    "nemo_conformer": NeMoConformerEngine,
    "indic_conformer": IndicConformerEngine,
}
TTS_ENGINES = {
    "emtiazz_vits": EmtiazzVitsEngine,
    "mms_tts": MMSTTSEngine,
    "indic_parler": IndicParlerTTSEngine,
}


def print_memory_table() -> None:
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — GPU memory table skipped.")
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(
        "GPU memory after loading all models: %.2f GB allocated | %.2f GB reserved | %.2f GB total (unified)",
        allocated, reserved, total,
    )


async def build_and_run() -> None:
    settings = Settings()

    import torch
    logger.info(
        "GPU: %s | CUDA available: %s",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        torch.cuda.is_available(),
    )

    mic = MicrophoneInput(sample_rate=settings.AUDIO_SAMPLE_RATE, device=settings.AUDIO_INPUT_DEVICE)
    speaker = SpeakerOutput(sample_rate=settings.AUDIO_SAMPLE_RATE, device=settings.AUDIO_OUTPUT_DEVICE)

    vad = SileroVAD(VADConfig(
        sample_rate=settings.AUDIO_SAMPLE_RATE,
        start_threshold=settings.VAD_START_THRESHOLD,
        end_silence_ms=settings.VAD_END_SILENCE_MS,
        min_speech_ms=settings.VAD_MIN_SPEECH_MS,
    ))

    logger.info("STT ইঞ্জিন লোড হচ্ছে (%s)...", settings.STT_ENGINE)
    stt_engine = STT_ENGINES[settings.STT_ENGINE](
        model_id=settings.STT_MODEL_ID, device=settings.STT_DEVICE, dtype=settings.STT_DTYPE,
        nemo_filename=settings.STT_NEMO_FILENAME, decoding=settings.STT_INDIC_DECODING,
    )
    await stt_engine.warmup()

    logger.info("TTS ইঞ্জিন লোড হচ্ছে (%s)...", settings.TTS_ENGINE)
    tts_engine = TTS_ENGINES[settings.TTS_ENGINE](
        device=settings.TTS_DEVICE, voice_description=settings.TTS_PARLER_VOICE_DESCRIPTION,
    )
    await tts_engine.warmup()
    speaker.sample_rate = tts_engine.sample_rate

    llm = OllamaLLMClient(
        model=settings.LLM_MODEL, host=settings.LLM_HOST,
        temperature=settings.LLM_TEMPERATURE, keep_alive=settings.LLM_KEEP_ALIVE,
        think=settings.LLM_THINK,
    )
    logger.info("LLM ওয়ার্মআপ হচ্ছে (%s)...", settings.LLM_MODEL)
    await llm.warmup()

    router = FastRouter(settings.LLM_NUM_PREDICT_GREETING, settings.LLM_NUM_PREDICT_NORMAL)

    retriever = None
    if settings.RAG_ENABLED:
        embed_cache = EmbeddingCache(
            model=settings.RAG_EMBED_MODEL, host=settings.LLM_HOST, cache_path=settings.RAG_CACHE_PATH
        )
        retriever = HybridRetriever(
            knowledge_dir=settings.RAG_KNOWLEDGE_DIR, embed_cache=embed_cache,
            chunk_chars=settings.RAG_CHUNK_CHARS, top_k=settings.RAG_TOP_K,
            min_score=settings.RAG_MIN_SCORE, w_embed=settings.RAG_WEIGHT_EMBED,
            w_tfidf=settings.RAG_WEIGHT_TFIDF, w_tfidf_word=settings.RAG_WEIGHT_TFIDF_WORD,
            w_tfidf_char=settings.RAG_WEIGHT_TFIDF_CHAR,
        )
        logger.info("RAG ইনডেক্স তৈরি হচ্ছে (%d passages)...", len(retriever.passages))
        await retriever.build_index()

    filler_cache = FillerCache(tts_engine, settings.filler_phrases_list)
    if settings.FILLER_ENABLED:
        logger.info("ফিলার ফ্রেজ প্রি-সিন্থেসাইজ হচ্ছে...")
        await filler_cache.prewarm()

    latency_logger = LatencyLogger(settings.LATENCY_LOG_PATH)
    print_memory_table()

    orchestrator = VoiceAssistantOrchestrator(
        mic=mic, speaker=speaker, vad=vad, stt_engine=stt_engine, llm=llm, router=router,
        tts_engine=tts_engine, filler_cache=filler_cache, retriever=retriever,
        latency_logger=latency_logger, stt_partial_interval_ms=settings.STT_PARTIAL_INTERVAL_MS,
        filler_trigger_ms=settings.FILLER_TRIGGER_MS, filler_enabled=settings.FILLER_ENABLED,
        barge_in_enabled=settings.BARGE_IN_ENABLED, rag_enabled=settings.RAG_ENABLED,
    )
    await orchestrator.run()


def main() -> None:
    try:
        asyncio.run(build_and_run())
    except KeyboardInterrupt:
        logger.info("বন্ধ হচ্ছে... (shutting down)")


if __name__ == "__main__":
    main()
