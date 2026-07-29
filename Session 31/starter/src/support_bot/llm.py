from functools import lru_cache

from langchain_openai import ChatOpenAI

from .config import MODEL_NAME, OPENAI_API_KEY, REASONING_EFFORT


@lru_cache(maxsize=1)
def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        reasoning_effort=REASONING_EFFORT,
        timeout=30,
        max_retries=2,
    )