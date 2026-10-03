"""bge-m3 embeddings via Ollama (`ollama pull bge-m3` — verified real,
multilingual, ~1.2GB), with a disk cache keyed by passage content hash so
re-embedding on every restart is unnecessary.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ollama


class EmbeddingCache:
    def __init__(self, model: str = "bge-m3", host: str = "http://localhost:11434",
                 cache_path: str = "knowledge/.embed_cache.json"):
        self.model = model
        self.client = ollama.AsyncClient(host=host)
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, List[float]] = {}
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    async def embed(self, text: str) -> np.ndarray:
        resp = await self.client.embeddings(model=self.model, prompt=text)
        return np.array(resp["embedding"], dtype=np.float32)

    async def embed_passages(self, items: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """items: list of (content_hash, text). Only embeds what's not already cached."""
        result: Dict[str, np.ndarray] = {}
        dirty = False
        for content_hash, text in items:
            if content_hash in self._cache:
                result[content_hash] = np.array(self._cache[content_hash], dtype=np.float32)
                continue
            vec = await self.embed(text)
            self._cache[content_hash] = vec.tolist()
            result[content_hash] = vec
            dirty = True
        if dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        return result
