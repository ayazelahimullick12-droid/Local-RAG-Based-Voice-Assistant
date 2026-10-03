# Main app container: STT (transformers) + TTS + LLM client + orchestrator.
#
# Based on the NGC PyTorch image so torch is a real CUDA build for the GB10
# (sm_121) GPU — do NOT `pip install torch/torchaudio` on top of this, it
# will silently replace the NGC build with a PyPI wheel that doesn't target
# this GPU. Research (Sept 2026) found NGC tags 26.04-26.08-py3 listed as
# live with aarch64 support, but couldn't confirm sm_121/Blackwell kernel
# presence from the catalog page alone — pull the LATEST available tag and
# verify with scripts/check_platform.py after build, not before.
ARG NGC_PYTORCH_TAG=26.08-py3
FROM nvcr.io/nvidia/pytorch:${NGC_PYTORCH_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-tts-coqui.txt requirements-tts-parler.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Installs the extra deps for the DEFAULT TTS_ENGINE (emtiazz_vits). If you
# switch TTS_ENGINE in config/.env, rebuild with the matching --build-arg,
# e.g. `--build-arg TTS_EXTRA=requirements-tts-parler.txt`, or install both.
ARG TTS_EXTRA=requirements-tts-coqui.txt
RUN pip install --no-cache-dir -r ${TTS_EXTRA}

COPY assistant/ ./assistant/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY knowledge/ ./knowledge/

ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "assistant.main"]
