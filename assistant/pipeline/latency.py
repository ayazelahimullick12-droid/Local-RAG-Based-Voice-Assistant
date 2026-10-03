"""Per-turn latency tracking: prints a one-line summary to the console and
appends a JSONL record, both timestamped from end-of-user-speech.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class TurnLatency:
    turn_id: int
    speech_end_ts: float = 0.0
    stt_final_ts: float = 0.0
    llm_first_token_ts: float = 0.0
    tts_first_audio_ts: float = 0.0
    playback_start_ts: float = 0.0
    filler_played: bool = False

    def summary(self) -> Dict[str, object]:
        base = self.speech_end_ts

        def rel_ms(ts: float):
            return round((ts - base) * 1000, 1) if ts else None

        return {
            "stt_final_ms": rel_ms(self.stt_final_ts),
            "llm_first_token_ms": rel_ms(self.llm_first_token_ts),
            "tts_first_audio_ms": rel_ms(self.tts_first_audio_ts),
            "total_to_audio_ms": rel_ms(self.playback_start_ts),
            "filler_played": self.filler_played,
        }


class LatencyLogger:
    def __init__(self, log_path: str = "logs/latency.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, turn: TurnLatency) -> None:
        summary = turn.summary()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"turn_id": turn.turn_id, "t": time.time(), **summary},
                                ensure_ascii=False) + "\n")
        print(
            f"[turn {turn.turn_id}] STT {summary['stt_final_ms']}ms | "
            f"LLM-1st-tok {summary['llm_first_token_ms']}ms | "
            f"TTS-1st-audio {summary['tts_first_audio_ms']}ms | "
            f"total-to-audio {summary['total_to_audio_ms']}ms"
            + (" | filler played" if summary["filler_played"] else "")
        )
