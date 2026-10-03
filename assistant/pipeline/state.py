from enum import Enum, auto


class ConversationState(Enum):
    LISTENING = auto()   # waiting for / buffering user speech
    THINKING = auto()     # STT finalized, LLM not yet producing audio
    SPEAKING = auto()     # assistant audio is queued/playing
