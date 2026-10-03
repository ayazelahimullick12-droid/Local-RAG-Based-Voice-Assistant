"""The asyncio producer/consumer pipeline tying everything together.

Concurrency model:
  - `run()`'s mic loop is the only reader of audio frames; it never blocks
    on a full response (response generation is a background task), so VAD
    keeps running — including while the assistant is talking, which is
    what makes barge-in possible.
  - `_generate_and_speak` streams LLM tokens, feeds them through the
    sentence chunker, and hands each phrase to a `tts_worker` task. TTS
    synthesis of chunk N+1 and playback of chunk N happen concurrently
    because `speaker.enqueue()` just queues audio — actual playback runs in
    SpeakerOutput's own background task (see audio/player.py).
  - Barge-in / stale-response safety uses a monotonic `_epoch` counter
    rather than relying purely on asyncio cancellation timing: every new
    response task captures the epoch at creation time, and every point
    where it would touch shared state (enqueue audio, flip back to
    LISTENING) re-checks that its epoch is still current first. This closes
    the race where a task's cleanup runs *after* barge-in has already
    started a new turn.

KNOWN LIMITATION: there is no acoustic echo cancellation. Without headphones
or a hardware-AEC USB speakerphone, the mic can pick up the assistant's own
voice and misfire barge-in. See README Troubleshooting.
"""
import asyncio
import logging
import time
from typing import List, Optional

from ..audio.transport import AudioInputTransport, AudioOutputTransport
from ..llm.client import OllamaLLMClient
from ..llm.prompts import NO_CONTEXT_REPLY_BN, SYSTEM_PROMPT_BN
from ..llm.router import FastRouter
from ..rag.retrieval import HybridRetriever
from ..stt.base import STTEngine
from ..stt.streaming import StreamingTranscriber
from ..text.chunker import SentenceChunker
from ..text.normalize import normalize_text
from ..tts.base import TTSEngine
from ..tts.filler_cache import FillerCache
from ..vad.silero_vad import SileroVAD, VADEvent
from .latency import LatencyLogger, TurnLatency
from .state import ConversationState

logger = logging.getLogger(__name__)


class VoiceAssistantOrchestrator:
    def __init__(self, *, mic: AudioInputTransport, speaker: AudioOutputTransport,
                 vad: SileroVAD, stt_engine: STTEngine, llm: OllamaLLMClient,
                 router: FastRouter, tts_engine: TTSEngine, filler_cache: FillerCache,
                 retriever: Optional[HybridRetriever], latency_logger: LatencyLogger,
                 stt_partial_interval_ms: int = 300, filler_trigger_ms: int = 700,
                 filler_enabled: bool = True, barge_in_enabled: bool = True,
                 rag_enabled: bool = True):
        self.mic = mic
        self.speaker = speaker
        self.vad = vad
        self.stt_engine = stt_engine
        self.llm = llm
        self.router = router
        self.tts_engine = tts_engine
        self.filler_cache = filler_cache
        self.retriever = retriever
        self.latency_logger = latency_logger
        self.filler_trigger_s = filler_trigger_ms / 1000
        self.filler_enabled = filler_enabled
        self.barge_in_enabled = barge_in_enabled
        self.rag_enabled = rag_enabled

        self.state = ConversationState.LISTENING
        self.streaming_stt = StreamingTranscriber(
            stt_engine, interval_ms=stt_partial_interval_ms, sample_rate=mic.sample_rate
        )
        self.history: List[dict] = []

        self._turn_id = 0
        self._epoch = 0
        self._response_task: Optional[asyncio.Task] = None
        self._speculative_context: Optional[str] = None
        self._speculative_task: Optional[asyncio.Task] = None

    async def run(self) -> None:
        await self.mic.start()
        await self.speaker.start()
        logger.info("শুনছি... (listening)")

        recording = False
        async for frame in self.mic.frames():
            event, _ = self.vad.process(frame)

            if self.state != ConversationState.LISTENING:
                # THINKING or SPEAKING: only react to a fresh speech start (barge-in).
                if self.barge_in_enabled and event == VADEvent.SPEECH_START:
                    await self._handle_barge_in()
                    recording = True
                    self.streaming_stt.start()
                    self.streaming_stt.append(frame)
                continue

            if event == VADEvent.SPEECH_START:
                recording = True
                self.streaming_stt.start()
            if recording:
                self.streaming_stt.append(frame)
                self._maybe_kick_speculative_rag()
            if event == VADEvent.SPEECH_END:
                recording = False
                await self._handle_utterance_end()

    # -- speculative retrieval on partial transcript -------------------------

    def _maybe_kick_speculative_rag(self) -> None:
        if not (self.rag_enabled and self.retriever):
            return
        text = self.streaming_stt.latest_partial
        if not text or text == self._speculative_context:
            return
        if len(text) - len(self._speculative_context or "") < 5:
            return
        self._speculative_context = text
        if self._speculative_task is None or self._speculative_task.done():
            self._speculative_task = asyncio.create_task(self.retriever.retrieve(text))

    # -- barge-in -------------------------------------------------------

    async def _handle_barge_in(self) -> None:
        logger.info("বার্জ-ইন: থামছি... (barge-in: stopping)")
        self._epoch += 1
        if self._response_task is not None:
            self._response_task.cancel()
        await self.speaker.clear()
        self.state = ConversationState.LISTENING

    # -- turn handling -------------------------------------------------------

    async def _handle_utterance_end(self) -> None:
        self._turn_id += 1
        self._epoch += 1
        my_epoch = self._epoch
        turn = TurnLatency(turn_id=self._turn_id, speech_end_ts=time.perf_counter())
        self.state = ConversationState.THINKING

        final = await self.streaming_stt.stop_and_finalize()
        turn.stt_final_ts = time.perf_counter()
        user_text = final.text.strip()
        if not user_text:
            self.state = ConversationState.LISTENING
            return
        logger.info("ব্যবহারকারী: %s", user_text)

        decision = self.router.route(user_text)

        retrieved = []
        if decision.route == "knowledge" and self.rag_enabled and self.retriever:
            if self._speculative_task is not None and not self._speculative_task.done():
                self._speculative_task.cancel()
            retrieved = await self.retriever.retrieve(user_text)

        messages = self._build_messages(user_text, retrieved, decision.route)

        # Fire-and-forget: the main mic loop must keep running so barge-in
        # can be detected while this generates/speaks.
        self._response_task = asyncio.create_task(
            self._generate_and_speak(
                messages, decision.num_predict, turn,
                is_greeting=(decision.route == "greeting"), my_epoch=my_epoch,
            )
        )

    def _build_messages(self, user_text: str, retrieved, route: str) -> List[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT_BN}]
        messages.extend(self.history[-6:])

        if route == "knowledge" and self.rag_enabled:
            if retrieved:
                context = "\n\n".join(f"[{r.passage.heading}]\n{r.passage.text}" for r in retrieved)
                user_text = (
                    f"প্রাসঙ্গিক তথ্য:\n{context}\n\n"
                    f"এই তথ্যের ভিত্তিতে বাংলায় সংক্ষেপে উত্তর দাও। প্রশ্ন: {user_text}"
                )
            elif self.retriever is not None and self.retriever.passages:
                user_text = (
                    f"{user_text}\n\n(প্রাসঙ্গিক কোনো তথ্য পাওয়া যায়নি। এই কথা বাংলায় বিনীতভাবে জানাও: "
                    f"\"{NO_CONTEXT_REPLY_BN}\")"
                )

        messages.append({"role": "user", "content": user_text})
        return messages

    # -- generation + speech -------------------------------------------------------

    async def _generate_and_speak(self, messages: List[dict], num_predict: int,
                                    turn: TurnLatency, is_greeting: bool, my_epoch: int) -> None:
        self.state = ConversationState.SPEAKING
        chunker = SentenceChunker(min_chars=15)
        audio_queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        first_token_seen = False
        first_audio_started = False
        filler_task: Optional[asyncio.Task] = None

        async def filler_watcher():
            await asyncio.sleep(self.filler_trigger_s)
            if my_epoch != self._epoch or first_audio_started or is_greeting:
                return
            if not self.filler_enabled:
                return
            filler = self.filler_cache.get_random()
            if filler is not None:
                turn.filler_played = True
                await self.speaker.enqueue(filler)

        async def tts_worker():
            nonlocal first_audio_started
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                if my_epoch != self._epoch:
                    continue
                pcm = await self.tts_engine.synthesize(normalize_text(chunk))
                if my_epoch != self._epoch:
                    continue
                if not first_audio_started:
                    turn.tts_first_audio_ts = time.perf_counter()
                    first_audio_started = True
                    if filler_task is not None and not filler_task.done():
                        filler_task.cancel()
                if not turn.playback_start_ts:
                    turn.playback_start_ts = time.perf_counter()
                await self.speaker.enqueue(pcm)

        if self.filler_enabled and not is_greeting:
            filler_task = asyncio.create_task(filler_watcher())
        worker_task = asyncio.create_task(tts_worker())

        full_reply = ""
        try:
            async for delta in self.llm.stream_chat(messages, num_predict=num_predict):
                if my_epoch != self._epoch:
                    break
                if not first_token_seen:
                    turn.llm_first_token_ts = time.perf_counter()
                    first_token_seen = True
                full_reply += delta
                for chunk in chunker.push(delta):
                    await audio_queue.put(chunk)
            else:
                for chunk in chunker.flush():
                    await audio_queue.put(chunk)
        finally:
            await audio_queue.put(None)
            await worker_task
            if filler_task is not None:
                filler_task.cancel()
            if my_epoch == self._epoch:
                self.state = ConversationState.LISTENING
                self.history.append({"role": "user", "content": messages[-1]["content"]})
                self.history.append({"role": "assistant", "content": full_reply})
                self.latency_logger.log(turn)
