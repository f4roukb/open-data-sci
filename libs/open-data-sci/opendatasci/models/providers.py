from enum import StrEnum, auto


class Provider(StrEnum):
    ANTHROPIC = auto()
    OPENAI = auto()
    BEDROCK = auto()
    GEMINI = auto()
    VERTEXAI = auto()
    AZURE = auto()
    OLLAMA = auto()
    VLLM = auto()
