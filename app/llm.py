"""Фабрика моделей: лёгкие задачи -> Qwen (Ollama), сложные -> Gemini."""
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from . import config

Role = Literal["light", "heavy"]


def _build(provider: str, model: str, temperature: float) -> BaseChatModel:
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Ключ берётся из переменной окружения GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    raise ValueError(f"Неизвестный провайдер модели: {provider}")


def get_llm(role: Role = "heavy", temperature: float = 0.0) -> BaseChatModel:
    """Вернуть модель по роли агента.

    light -> аналитик заявки (извлечение полей),
    heavy -> supervisor, проверщик соответствия, составитель КП.
    """
    if role == "light":
        return _build(config.LLM_LIGHT_PROVIDER, config.LLM_LIGHT_MODEL, temperature)
    return _build(config.LLM_HEAVY_PROVIDER, config.LLM_HEAVY_MODEL, temperature)


def get_embeddings():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
