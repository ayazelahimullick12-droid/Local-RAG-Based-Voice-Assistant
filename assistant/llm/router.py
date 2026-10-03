"""Fast router: greetings/small talk skip RAG entirely and use a short
token budget, so those turns never pay retrieval latency. Everything else
is treated as a knowledge question and goes through RAG retrieval.
"""
import re
from dataclasses import dataclass

GREETING_PATTERNS = [
    "হ্যালো", "হাই", "আসসালামু আলাইকুম", "সালাম", "কেমন আছেন", "কেমন আছো",
    "ধন্যবাদ", "শুকরিয়া", "গুড মর্নিং", "শুভ সকাল", "শুভ বিকাল", "শুভ রাত্রি",
    "বিদায়", "আল্লাহ হাফেজ", "ভালো আছি", "আচ্ছা ঠিক আছে",
]

_GREETING_MAX_CHARS = 20


@dataclass
class RouteDecision:
    route: str  # "greeting" | "knowledge"
    num_predict: int


class FastRouter:
    def __init__(self, num_predict_greeting: int = 40, num_predict_normal: int = 120):
        self.num_predict_greeting = num_predict_greeting
        self.num_predict_normal = num_predict_normal
        self._pattern = re.compile("|".join(re.escape(p) for p in GREETING_PATTERNS))

    def route(self, text: str) -> RouteDecision:
        stripped = text.strip()
        is_short_greeting = len(stripped) <= _GREETING_MAX_CHARS and bool(self._pattern.search(stripped))
        if is_short_greeting:
            return RouteDecision(route="greeting", num_predict=self.num_predict_greeting)
        return RouteDecision(route="knowledge", num_predict=self.num_predict_normal)
