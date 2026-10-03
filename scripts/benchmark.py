"""Feeds sample Bengali WAV files through the pipeline WITHOUT the
microphone, and prints per-stage latency: STT final, LLM first token, TTS
first audio, total. Use --stt/--tts to A-B different engine combinations
against the same recordings.

    python -m scripts.benchmark --samples benchmark_samples --stt bangla_asr --tts emtiazz_vits
"""
import argparse
import asyncio
import glob
import time

import soundfile as sf

from assistant.config import Settings
from assistant.llm.client import OllamaLLMClient
from assistant.llm.prompts import SYSTEM_PROMPT_BN
from assistant.llm.router import FastRouter
from assistant.main import STT_ENGINES, TTS_ENGINES
from assistant.text.chunker import SentenceChunker
from assistant.text.normalize import normalize_text


async def run_one(wav_path: str, stt_engine, llm: OllamaLLMClient, router: FastRouter, tts_engine) -> dict:
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{wav_path}: expected 16kHz, got {sr}Hz — resample the sample set first")

    t0 = time.perf_counter()
    stt_result = await stt_engine.transcribe(audio, sr)
    t_stt = time.perf_counter()

    decision = router.route(stt_result.text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_BN},
        {"role": "user", "content": stt_result.text},
    ]

    chunker = SentenceChunker(min_chars=15)
    t_llm_first = None
    t_tts_first = None
    async for delta in llm.stream_chat(messages, num_predict=decision.num_predict):
        if t_llm_first is None:
            t_llm_first = time.perf_counter()
        for chunk in chunker.push(delta):
            if t_tts_first is None:
                await tts_engine.synthesize(normalize_text(chunk))
                t_tts_first = time.perf_counter()
    t_end = time.perf_counter()

    def rel_ms(t):
        return round((t - t0) * 1000, 1) if t else None

    return {
        "file": wav_path,
        "transcript": stt_result.text,
        "stt_ms": rel_ms(t_stt),
        "llm_first_token_ms": rel_ms(t_llm_first),
        "tts_first_audio_ms": rel_ms(t_tts_first),
        "total_ms": rel_ms(t_end),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="benchmark_samples")
    parser.add_argument("--stt", default=None, help="override STT_ENGINE from config/.env")
    parser.add_argument("--tts", default=None, help="override TTS_ENGINE from config/.env")
    args = parser.parse_args()

    settings = Settings()
    stt_name = args.stt or settings.STT_ENGINE
    tts_name = args.tts or settings.TTS_ENGINE

    print(f"Loading STT={stt_name}, TTS={tts_name}, LLM={settings.LLM_MODEL}...")
    stt_engine = STT_ENGINES[stt_name](
        model_id=settings.STT_MODEL_ID, device=settings.STT_DEVICE, dtype=settings.STT_DTYPE,
        nemo_filename=settings.STT_NEMO_FILENAME, decoding=settings.STT_INDIC_DECODING,
    )
    await stt_engine.warmup()
    tts_engine = TTS_ENGINES[tts_name](
        device=settings.TTS_DEVICE, voice_description=settings.TTS_PARLER_VOICE_DESCRIPTION,
    )
    await tts_engine.warmup()
    llm = OllamaLLMClient(
        model=settings.LLM_MODEL, host=settings.LLM_HOST, temperature=settings.LLM_TEMPERATURE,
        keep_alive=settings.LLM_KEEP_ALIVE, think=settings.LLM_THINK,
    )
    await llm.warmup()
    router = FastRouter(settings.LLM_NUM_PREDICT_GREETING, settings.LLM_NUM_PREDICT_NORMAL)

    files = sorted(glob.glob(f"{args.samples}/*.wav"))
    if not files:
        print(f"No .wav files found in {args.samples}/ — drop 16kHz mono Bengali test "
              f"recordings there (include some natural conversational speech, not just "
              f"clean read speech).")
        return

    print(f"\n{'file':30s} {'stt_ms':>8s} {'llm_1st_ms':>11s} {'tts_1st_ms':>11s} {'total_ms':>9s}  transcript")
    for f in files:
        r = await run_one(f, stt_engine, llm, router, tts_engine)
        print(f"{r['file']:30s} {str(r['stt_ms']):>8s} {str(r['llm_first_token_ms']):>11s} "
              f"{str(r['tts_first_audio_ms']):>11s} {str(r['total_ms']):>9s}  {r['transcript'][:40]}")


if __name__ == "__main__":
    asyncio.run(main())
