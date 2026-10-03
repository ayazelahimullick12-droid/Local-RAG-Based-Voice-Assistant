"""Platform sanity check for the DGX Spark box. Run this FIRST, before
anything else, and read every line of output — don't assume any of this
passes just because the container started.

    python -m scripts.check_platform
"""
import json
import platform
import subprocess
import sys
import urllib.request


def check_arch() -> None:
    machine = platform.machine()
    print(f"[platform] machine={machine} system={platform.system()} python={sys.version.split()[0]}")
    if machine not in ("aarch64", "arm64"):
        print("  WARNING: expected aarch64 on DGX Spark — this doesn't look like that box.")


def check_nvidia_smi() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"[nvidia-smi] {out.stdout.strip() or out.stderr.strip()}")
    except FileNotFoundError:
        print("[nvidia-smi] not found on PATH")


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError:
        print("[torch] NOT INSTALLED — run this inside the nvcr.io/nvidia/pytorch container.")
        return

    print(f"[torch] version={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  FAIL: CUDA not visible to torch. Check `docker run --gpus all` / `runtime: nvidia`.")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[torch] device={name} capability=sm_{cap[0]}{cap[1]}")
    if f"{cap[0]}{cap[1]}" != "121":
        print(f"  NOTE: expected sm_121 for GB10 Blackwell — got sm_{cap[0]}{cap[1]}. "
              f"Confirm your NGC container tag actually ships Blackwell kernels.")

    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"[torch] matmul smoke test OK, result checksum={float(y.sum()):.3e}")


def check_ctranslate2() -> None:
    try:
        import ctranslate2
    except ImportError:
        print("[ctranslate2] not installed (only relevant if experimenting with faster-whisper)")
        return
    try:
        count = ctranslate2.get_cuda_device_count()
        print(f"[ctranslate2] version={ctranslate2.__version__} cuda_device_count={count}")
        if count == 0:
            print(
                "  WARNING: ctranslate2 sees 0 CUDA devices — it will silently run on CPU. "
                "There is no official aarch64+CUDA wheel as of this writing (confirmed via "
                "speaches-ai/speaches#620, opened specifically about DGX Spark/Grace); "
                "faster-whisper stays CPU-only here unless you build ctranslate2 from source "
                "with CUDA enabled. This project does not depend on faster-whisper by default."
            )
    except Exception as e:
        print(f"  ERROR checking ctranslate2 CUDA support: {e}")


def check_ollama() -> None:
    host = "http://localhost:11434"
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"[ollama] reachable at {host}, {len(models)} model(s) pulled: {models}")
        expected = {"gemma4:26b", "bge-m3"}
        missing = expected - set(models)
        if missing:
            print(f"  NOTE: expected models not found — run `ollama pull <name>` for: {sorted(missing)}")
    except Exception as e:
        print(f"[ollama] NOT reachable at {host}: {e}")


def check_audio() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("[audio] sounddevice not installed")
        return
    try:
        devices = sd.query_devices()
        inputs = [d["name"] for d in devices if d["max_input_channels"] > 0]
        outputs = [d["name"] for d in devices if d["max_output_channels"] > 0]
        print(f"[audio] input devices: {inputs}")
        print(f"[audio] output devices: {outputs}")
        if not inputs or not outputs:
            print("  WARNING: no input or output device found — check /dev/snd passthrough "
                  "(and PulseAudio/PipeWire socket mount if the host uses one, not raw ALSA).")
    except Exception as e:
        print(f"  ERROR querying audio devices: {e}")


if __name__ == "__main__":
    print("=== DGX Spark platform check ===")
    check_arch()
    check_nvidia_smi()
    check_torch_cuda()
    check_ctranslate2()
    check_ollama()
    check_audio()
    print("=== done — re-run after each config change to confirm nothing silently fell back to CPU ===")
