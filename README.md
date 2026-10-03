# Bangla Voice Assistant — v1.0

A fully local, real-time Bengali voice assistant: microphone in, speakers
out, no cloud calls, no web UI. Built for the MSI EdgeXpert MS-C931 (NVIDIA
DGX Spark, GB10 Grace Blackwell, ARM64, 128GB unified memory).

**Status: this is a code scaffold, not a verified build.** It was written
on a Windows dev machine with no access to the actual DGX Spark hardware —
every model ID, package name, and API call below was checked against live
sources (Hugging Face, Ollama library, PyPI, GitHub issues) as of September
2026, but *none of it has been run on the target GPU yet*. Follow the
"Bring-up order" section below exactly — it's the same incremental order
the original spec asked for, and it's how you'll catch the things that only
show up on real aarch64 + Blackwell hardware.

## Architecture choice: custom asyncio pipeline (not Pipecat/LiveKit)

Pipecat and LiveKit Agents are both built around a transport abstraction
aimed at WebRTC/telephony — rooms, participants, SFU connections. For a
single local mic-and-speakers loop, that pulls in a stack of networking
dependencies (aiortc, livekit-server-sdk, etc.) that this project never
uses, and their pipeline frameworks add a layer of indirection between
"model finished a chunk" and "chunk is playing" that makes the barge-in
cancellation semantics harder to reason about than plain `asyncio.Task`
cancellation.

A hand-rolled `asyncio` producer/consumer pipeline (`assistant/pipeline/orchestrator.py`)
gives full control over exactly when GPU work happens, exactly when audio
is queued vs. played, and exactly how a barge-in interrupts an in-flight
turn — which matters a lot for hitting the <1s latency target. The
`AudioInputTransport`/`AudioOutputTransport` interfaces
(`assistant/audio/transport.py`) are the deliberate seam for adding a
WebRTC transport (e.g. a LiveKit room) later without touching the
pipeline, STT, LLM, TTS, or RAG code at all.

## Project layout

```
assistant/
  config.py            all settings (loads config/.env)
  audio/                transport interfaces + sounddevice mic/speaker
  vad/                  Silero VAD + speech start/end state machine
  stt/                  BanglaASR / NeMo Conformer / Indic Conformer + streaming wrapper
  llm/                  Ollama streaming client + fast router + Bengali prompts
  tts/                  EMTIAZZ VITS / MMS-TTS / Indic Parler-TTS + filler cache
  text/                 sentence chunker + number/date/currency normalization
  rag/                  markdown chunking, bge-m3 embeddings, hybrid retrieval
  pipeline/             the orchestrator, conversation state, latency logging
  main.py                entry point
scripts/
  check_platform.py     GPU/CUDA/ollama/audio sanity check — run this first
  benchmark.py           per-stage latency over sample WAVs, no mic needed
docker/                  Dockerfile.app, Dockerfile.nemo, docker-compose.yml
knowledge/               your .md/.txt RAG source files
config/.env.example       copy to config/.env and edit
```

## Model choices & verification status

Checked live (HF model cards, ollama.com/library, PyPI, GitHub) during
scaffolding — not run on GPU yet. **Bold** = differs from the original
assumption.

| Component | Model | License | Verified? |
|---|---|---|---|
| VAD | `silero-vad` (pip package, snakers4/silero-vad) | MIT | ✅ exists, current install path confirmed |
| STT primary | `bangla-speech-processing/BanglaASR` | MIT | ✅ exists, standard Whisper loading, no quirks found |
| STT fallback | `kazalbrur/bangla-stt-conformer-120m-dialects` | **Apache-2.0** (not CC-BY-NC-4.0) | ✅ exists, needs NeMo |
| STT fallback | `ai4bharat/indic-conformer-600m-multilingual` | MIT | ✅ exists, **no NeMo needed** — plain transformers |
| LLM primary | `gemma4:26b` (Ollama) | Apache-2.0 | ✅ **real** — Gemma 4 shipped Apr 2026, 25.2B total/3.8B active MoE, ~19GB |
| LLM A/B | `qwen3.5:35b-a3b`, `qwen3.6:35b-a3b` (Ollama) | — | ✅ exact tags confirmed live |
| Embeddings | `bge-m3` (Ollama) | MIT | ✅ confirmed pullable |
| TTS primary | `EMTIAZZ/bangladeshi-bangla-tts-vits` | Apache-2.0 | ✅ exists, **loads via Coqui `TTS` package, not transformers** |
| TTS fallback | `facebook/mms-tts-ben` | **CC-BY-NC-4.0 (non-commercial)** | ✅ confirmed, 16kHz, standard `VitsModel` |
| TTS fallback | `ai4bharat/indic-parler-tts` | Apache-2.0, **0.9B** params | ✅ exists, Arjun/Aditi voices confirmed |
| faster-whisper | CTranslate2 GPU on aarch64 | — | ❌ **no official CUDA wheel for aarch64** — confirmed via `speaches-ai/speaches#620`. Not used by default; see Troubleshooting if you want to attempt a source build. |
| Orpheus Bangla TTS | several community repos (`asif00/orpheus-bangla-tts`, etc.) | mixed | Found but fragmented / base model gated — **not implemented**, per the spec's "if not, drop them." |

## Setup on DGX Spark

1. **Platform check first, before touching any model code:**
   ```bash
   docker run --rm --runtime=nvidia --gpus all nvcr.io/nvidia/pytorch:26.08-py3 \
     python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
   ```
   Confirm this prints `True` and a capability that corresponds to GB10 Blackwell before building anything on top.

2. **Ollama** (bare-metal or container with GPU access):
   ```bash
   ollama pull gemma4:26b
   ollama pull bge-m3
   # A/B candidates, optional:
   ollama pull gemma4:e4b
   ollama pull qwen3.6:35b-a3b
   ```

3. **Build the app container:**
   ```bash
   docker compose -f docker/docker-compose.yml up --build
   ```
   or run bare-metal inside an `nvcr.io/nvidia/pytorch` container with
   `pip install -r requirements.txt`. TTS engine deps are split out per
   engine (`requirements-tts-coqui.txt` for the default `emtiazz_vits`,
   `requirements-tts-parler.txt` for `indic_parler`) — each is imported
   lazily, and neither is required to run the other, so a failure
   installing one doesn't block the rest of the app. Install whichever
   matches your `TTS_ENGINE`.

4. **Copy and edit config:**
   ```bash
   cp config/.env.example config/.env
   ```

5. **Run the platform check inside the container:**
   ```bash
   python -m scripts.check_platform
   ```
   Fix everything it flags before proceeding — especially "CUDA not
   available" and "0 CUDA devices" for any component.

## Bring-up order (do not skip steps)

This mirrors the original build plan. Each step should be verified with
real output before moving to the next — this scaffold has not done that,
so treat every step as "first time running this."

1. `python -m scripts.check_platform` — GPU, CUDA capability, Ollama reachability, audio devices.
2. STT alone: `python -m scripts.benchmark --stt bangla_asr` against one sample WAV, confirm the transcript is sane and check the reported `stt_ms`.
3. TTS alone: instantiate the TTS engine directly and listen to the output (see Troubleshooting for the EMTIAZZ loader, which is the least-verified piece in this codebase).
4. LLM streaming: `ollama run gemma4:26b` manually first to confirm the model loads and generates Bengali; then exercise `OllamaLLMClient.stream_chat` and confirm `think=False` actually suppresses reasoning output (check for a `<think>` tag leaking through — see the config comment on `LLM_THINK`).
5. Full loop, mic off (`scripts/benchmark.py` end-to-end): confirm per-stage latency numbers.
6. Full loop, live mic (`python -m assistant.main`): confirm the five latency optimizations behave (see below), then confirm barge-in.
7. RAG: drop real documents in `knowledge/`, confirm retrieval quality and the <100ms retrieval budget.

## Configuration

Every tunable is in `config/.env` (see `config/.env.example` for the full,
commented list) — VAD thresholds, STT/TTS/LLM engine switches, filler
toggle and phrases, RAG weights, barge-in toggle.

## The five latency optimizations — where they live

1. **Streaming transcription** — `assistant/stt/streaming.py`, re-transcribes the growing buffer every `STT_PARTIAL_INTERVAL_MS` (default 300ms).
2. **Sentence chunking** — `assistant/text/chunker.py`, splits on `। ? ! , ; \n` with a 15-char minimum, sends the first chunk as soon as it's ready.
3. **Parallel execution** — `assistant/pipeline/orchestrator.py`: LLM generation, TTS synthesis of the next chunk, and playback of the current chunk run as concurrent asyncio tasks; GPU calls go through `asyncio.to_thread` so the event loop never blocks. Barge-in uses a monotonic epoch counter (see the module docstring) to safely cancel an in-flight response and clear the audio queue without races.
4. **Bypass thinking** — `assistant/llm/router.py` (greeting fast-path, short `num_predict`) + `LLM_THINK=false` in `assistant/llm/client.py`.
5. **Proactive fillers** — `assistant/tts/filler_cache.py` (pre-synthesized at startup) + the `filler_watcher` task in the orchestrator (plays only if no real audio within `FILLER_TRIGGER_MS`, never on greetings, never repeats).

## Memory usage — ESTIMATED, not measured

No hardware access during scaffolding, so these are rough figures from each
model's reported size, not a real `nvidia-smi` snapshot. **Run
`python -m assistant.main` and read the "GPU memory after loading all
models" line it prints (via `print_memory_table()` in `assistant/main.py`)
for real numbers, and put them here.**

| Component | Rough size |
|---|---|
| BanglaASR (Whisper-small, bf16) | ~0.5–1 GB |
| EMTIAZZ VITS | ~0.2–0.5 GB |
| gemma4:26b (via Ollama, separate process) | ~19 GB |
| bge-m3 (via Ollama, separate process) | ~1.2 GB |
| Silero VAD | CPU only, negligible |
| **Estimated total** | **~21–23 GB**, vs. 128 GB unified — comfortable margin |

## Tuning guide

- `VAD_END_SILENCE_MS` (default 500): lower = snappier turn-taking but more
  risk of cutting off a user mid-thought during natural pauses.
- `STT_PARTIAL_INTERVAL_MS` (default 300): lower = more responsive partial
  transcripts and RAG speculation, at the cost of more GPU STT calls per
  utterance.
- `FILLER_TRIGGER_MS` (default 700): should sit comfortably below your
  measured p50 total-to-audio latency but above your p90, so fillers are
  rare on a good day and common on a slow one.
- `RAG_MIN_SCORE` (default 0.15): raise if the assistant is citing
  irrelevant passages, lower if it's saying "I don't have that information"
  too often on questions your knowledge/ docs actually answer.
- `LLM_NUM_PREDICT_NORMAL` / `_GREETING`: raise `_NORMAL` if answers get cut
  off mid-sentence; the system prompt already asks for 1-3 short sentences,
  so 120 tokens should rarely bind.

## Known limitations — read before relying on this

- **No acoustic echo cancellation.** Barge-in detection runs on raw VAD
  over the mic signal; without headphones or a hardware-AEC USB
  speakerphone (e.g. Jabra/Anker conference units), the mic can pick up the
  assistant's own voice through the speakers and falsely trigger a
  barge-in. This is the single biggest practical risk to the barge-in
  feature working well and was out of scope to solve in this pass — adding
  `webrtc-audio-processing` or a similar AEC library is the natural next
  step if this turns out to be a real problem on your setup.
- **Bengali number-word table is unverified.** `assistant/text/normalize.py`
  has a 0-99 word table written from general knowledge, not checked against
  a native-speaker reference. Spot-check it, especially 50-99.
- **EMTIAZZ VITS checkpoint/config filenames are best-guess.** The loader
  tries `pytorch_model.pth` / `config.json` and falls back to scanning the
  repo file listing — confirm against the live repo if it fails to load.
- **NeMo `.nemo` checkpoint filename is auto-detected**, not hardcoded, for
  the same reason — set `STT_NEMO_FILENAME` explicitly if it picks the
  wrong file in a repo with multiple checkpoints.
- **`mms-tts-ben` is CC-BY-NC-4.0** — non-commercial. Don't ship it as a
  fallback in anything commercial without checking that license again.

## Troubleshooting

**"CUDA not available" / everything silently runs on CPU**
Confirm `docker run --gpus all` or `runtime: nvidia` in compose is actually
taking effect — `nvidia-smi` should work *inside* the container. This is
the single most common way ARM64 GPU work silently degrades.

**`torch.cuda.get_device_capability()` doesn't say sm_121 / matmul is slow**
Your NGC container tag may predate Blackwell/GB10 kernel support. Try the
latest available `nvcr.io/nvidia/pytorch` tag; forum reports during
research pointed at 25.10/25.11 as a practical minimum, with 26.04-26.08
confirmed live on the catalog as of Sept 2026 — always check for something
newer than what's written here.

**faster-whisper / ctranslate2 runs on CPU even though you asked for CUDA**
Expected — there is no official aarch64+CUDA `ctranslate2` wheel (confirmed
via `speaches-ai/speaches#620`, opened specifically about this DGX
Spark/Grace scenario). Either build CTranslate2 from source against an NGC
CUDA devel image, or don't use `faster_whisper` — it isn't wired into
`STT_ENGINE` by default for exactly this reason.

**`nemo_toolkit` install fights with `transformers`/`torch` versions**
Don't install it in the main app container. Use `docker/Dockerfile.nemo`
(a separate NGC NeMo base image) for the `nemo_conformer` fallback STT.

**`coqui-tts` (EMTIAZZ loader) won't install, or `TTS` import fails**
The upstream `coqui-ai/TTS` package is semi-abandoned and picky about
Python versions; try the `coqui-tts` community fork listed in
`requirements.txt` first. If both fail, pin `TTS==0.22.0` (the version the
EMTIAZZ training repo's own `requirements.txt` uses) and match its Python
version constraint.

**No sound in/out inside the container**
DGX OS (Ubuntu-based) may run PipeWire/PulseAudio rather than raw ALSA.
`/dev/snd` passthrough (already in `docker-compose.yml`) covers plain ALSA;
if that's not enough, also mount the Pulse socket and set `PULSE_SERVER`
(see the comment at the top of `docker/docker-compose.yml`).

**`think=False` doesn't suppress reasoning output**
Check your `ollama` / `ollama-python` version — there are open upstream
issues (`ollama/ollama-python#529`) about inconsistent handling of the
`think` kwarg across versions. Confirm with `ollama run gemma4:26b --think=false`
on the CLI first to isolate whether it's a server-side or client-side issue.

**Barge-in feels laggy or doesn't fire**
Check `VAD_START_THRESHOLD` and `VAD_MIN_SPEECH_MS` first — they gate how
fast a genuine speech onset is recognized. Then check the echo-cancellation
limitation above; a false negative (assistant doesn't stop) is more likely
to be echo suppression accidentally working *against* you (mic gain too
low picking up user's voice) while a false positive (assistant stops for
no reason) is almost always the AEC gap.
