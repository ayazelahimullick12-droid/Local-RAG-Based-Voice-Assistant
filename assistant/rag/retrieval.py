"""Hybrid retriever: 0.65 * bge-m3 cosine similarity + 0.35 * (0.6 * word
TF-IDF + 0.4 * char-trigram TF-IDF). Discards passages below 0.15,
returns the top 3. All in-memory numpy ops after the index is built, so a
single retrieve() call should stay well under the 100ms budget for a
knowledge/ folder of reasonable size.
"""
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .chunking import Passage, load_passages
from .embeddings import EmbeddingCache
from .tfidf import HybridTfidf


@dataclass
class RetrievedPassage:
    passage: Passage
    score: float


class HybridRetriever:
    def __init__(self, knowledge_dir: str, embed_cache: EmbeddingCache,
                 chunk_chars: int = 900, top_k: int = 3, min_score: float = 0.15,
                 w_embed: float = 0.65, w_tfidf: float = 0.35,
                 w_tfidf_word: float = 0.6, w_tfidf_char: float = 0.4):
        self.top_k = top_k
        self.min_score = min_score
        self.w_embed = w_embed
        self.w_tfidf = w_tfidf
        self.embed_cache = embed_cache

        self.passages: List[Passage] = load_passages(knowledge_dir, chunk_chars)
        self.tfidf = HybridTfidf([p.text for p in self.passages], w_tfidf_word, w_tfidf_char)
        self._embeddings: Optional[np.ndarray] = None

    async def build_index(self) -> None:
        if not self.passages:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            return
        items = [(p.content_hash, p.text) for p in self.passages]
        vecs = await self.embed_cache.embed_passages(items)
        self._embeddings = np.stack([vecs[p.content_hash] for p in self.passages])

    async def retrieve(self, query: str) -> List[RetrievedPassage]:
        if not self.passages or not query.strip():
            return []
        query_vec = await self.embed_cache.embed(query)
        embed_sim = _cosine_matrix(self._embeddings, query_vec)
        tfidf_sim = self.tfidf.score(query)
        combined = self.w_embed * embed_sim + self.w_tfidf * tfidf_sim

        order = np.argsort(-combined)[: self.top_k]
        return [
            RetrievedPassage(passage=self.passages[i], score=float(combined[i]))
            for i in order if combined[i] >= self.min_score
        ]


def _cosine_matrix(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros(0)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    denom = matrix_norm * query_norm
    denom[denom == 0] = 1e-9
    return (matrix @ query) / denom
