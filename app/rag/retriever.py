"""Поиск по базе знаний и готовый инструмент для агентов."""
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.tools import tool

from .. import config
from ..llm import get_embeddings
from .ingest import COLLECTION


@lru_cache(maxsize=1)
def _store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def search_kb(query: str, k: int = 3) -> list[dict]:
    docs = _store().similarity_search(query, k=k)
    return [
        {"text": d.page_content, "source": d.metadata.get("source"), "section": d.metadata.get("section")}
        for d in docs
    ]


@tool
def kb_search(query: str) -> str:
    """Поиск по базе знаний: выбор типа велосипеда, подбор размера рамы и колёс,
    глоссарий характеристик, частые вопросы.
    Не содержит цен, наличия, условий доставки, оплаты, гарантии и скидок."""
    hits = search_kb(query)
    if not hits:
        return "В базе знаний ничего не найдено."
    return "\n\n---\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)
