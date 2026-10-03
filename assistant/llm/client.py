"""Streaming Ollama chat client.

Default model gemma4:26b — verified real and current (Google released
Gemma 4 in April 2026; the 26b Ollama tag is 25.2B total / 3.8B active MoE
params, ~19GB, which matches the spec's target almost exactly). A/B
candidates: qwen3.5:35b-a3b, qwen3.6:35b-a3b, gemma4:e4b (all verified live
Ollama library tags as of Sept 2026).

`think=False` is passed as a top-level kwarg to chat() — verified as the
correct location for ollama-python, though some versions have had
inconsistent handling (ollama/ollama-python#529); if replies start
including <think> content, check your installed ollama-python version.
"""
from typing import AsyncIterator, Dict, List

import ollama


class OllamaLLMClient:
    def __init__(self, model: str, host: str = "http://localhost:11434",
                 temperature: float = 0.6, keep_alive: str = "-1", think: bool = False):
        self.model = model
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.think = think
        self.client = ollama.AsyncClient(host=host)

    async def warmup(self) -> None:
        async for _ in self.stream_chat([{"role": "user", "content": "হ্যালো"}], num_predict=8):
            pass

    async def stream_chat(self, messages: List[Dict[str, str]], num_predict: int) -> AsyncIterator[str]:
        stream = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            think=self.think,
            keep_alive=self.keep_alive,
            options={
                "temperature": self.temperature,
                "num_predict": num_predict,
            },
        )
        async for part in stream:
            content = part.get("message", {}).get("content", "")
            if content:
                yield content
