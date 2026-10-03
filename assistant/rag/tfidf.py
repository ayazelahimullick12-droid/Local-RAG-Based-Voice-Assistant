"""Word-level + character-trigram TF-IDF, combined per RAG_WEIGHT_TFIDF_*.
Character-trigrams help with Bengali morphology/spelling variation that
word-level matching alone would miss.
"""
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class HybridTfidf:
    def __init__(self, texts: List[str], word_weight: float = 0.6, char_weight: float = 0.4):
        self.word_weight = word_weight
        self.char_weight = char_weight
        self.word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b")
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
        self.word_matrix = self.word_vec.fit_transform(texts) if texts else None
        self.char_matrix = self.char_vec.fit_transform(texts) if texts else None

    def score(self, query: str) -> np.ndarray:
        if self.word_matrix is None:
            return np.zeros(0)
        word_sim = _cosine(self.word_matrix, self.word_vec.transform([query]))
        char_sim = _cosine(self.char_matrix, self.char_vec.transform([query]))
        return self.word_weight * word_sim + self.char_weight * char_sim


def _cosine(matrix, query_vec) -> np.ndarray:
    matrix_norm = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.flatten()
    query_norm = float(np.sqrt(query_vec.multiply(query_vec).sum()))
    dot = (matrix @ query_vec.T).toarray().flatten()
    denom = matrix_norm * query_norm
    denom[denom == 0] = 1e-9
    return dot / denom
