"""Состояние графа. Живёт в чекпоинтере по thread_id (краткосрочная память)."""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict, total=False):
    # диалог и клиент
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    profile: dict  # долговременная память о клиенте

    # результаты агентов
    requirements: dict  # переносится между репликами диалога
    intent: str
    products: list[dict[str, Any]]    # всё, что вернул каталог
    candidates: list[dict[str, Any]]  # прошли жёсткие требования
    rejected: list[dict[str, Any]]    # отсеяны: {"product", "reasons"}
    checks: list[dict[str, Any]]
    ranked: list[dict[str, Any]]      # подходящие, отсортированы
    kb_notes: str                     # ответ консультанта (RAG)
    search_meta: dict
    answer: str                       # итоговый ответ этой реплики

    # служебное для supervisor
    analyzed: bool
    searched: bool
    checked: bool
    advised: bool
    written: bool
    steps: int
    route: str
    route_reason: str
    awaiting: bool      # прошлая реплика ассистента была уточняющим вопросом (живёт между репликами)
    child_asked: bool   # вопрос о возрасте или росте ребёнка уже задавался в этой заявке
    req_note: str       # откуда требования: новая заявка / уточнение / профиль
