"""Агент-аналитик (лёгкая модель, Qwen): текст заявки -> структурированные требования."""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..logic import text_of
from ..memory import profile_to_text
from ..schemas import Requirements

SYSTEM = """Ты аналитик заявок магазина велосипедов. Разбери последнее сообщение клиента
и верни требования строго по схеме.

Правила:
- intent: "order": клиент хочет подобрать или купить велосипед; "consult": вопрос о выборе,
  размерах, доставке, оплате, гарантии без заказа; "other": приветствие или не по теме.
- category: mountain (горный), road (шоссейный), city (городской), hybrid (гибридный),
  kids (детский). Если из сообщения и предыдущих требований тип неясен, null.
- quantity: сколько велосипедов, по умолчанию 1.
- max_price: максимум за ОДИН велосипед в рублях, числом. «до 20 тысяч» = 20000.
  Если назван общий бюджет на несколько штук, раздели его на количество.
- wheel_size: диаметр колёс в дюймах строкой, например "29".
- must_have: обязательные требования клиента («обязательно дисковые тормоза»).
- wishes: необязательные пожелания («недорогой», «лёгкий»).
- question: только если intent="order" и тип велосипеда определить нельзя: ОДИН короткий
  уточняющий вопрос. Иначе null.
- Если даны предыдущие требования, сохрани их и измени только то, что клиент изменил
  в новом сообщении. Данные о клиенте из прошлых обращений используй, только если
  он прямо ссылается на прошлый заказ («как в прошлый раз»).
- Ничего не выдумывай: чего нет в сообщении, оставляй пустым."""

OTHER_REPLY = (
    "Я помогаю подобрать велосипед и подготовить коммерческое предложение. "
    "Опишите, что вам нужно: тип велосипеда, бюджет, диаметр колёс, количество. "
    "Также могу ответить на вопросы о размерах, доставке, оплате и гарантии."
)
ASK_CATEGORY = "Какой тип велосипеда вам нужен: горный, шоссейный, городской, гибридный или детский?"


def _history(messages: list, limit: int = 6) -> str:
    lines = []
    for m in messages[-limit - 1 : -1]:  # без последнего: оно показывается отдельно
        role = "Клиент" if getattr(m, "type", "") == "human" else "Ассистент"
        lines.append(f"{role}: {text_of(m.content)[:400]}")
    return "\n".join(lines) or "(начало диалога)"


async def analyst_node(state: dict) -> dict:
    messages = state.get("messages", [])
    last = text_of(messages[-1].content) if messages else ""
    previous = state.get("requirements") or {}

    prompt = (
        f"Предыдущие требования (JSON): {json.dumps(previous, ensure_ascii=False) if previous else 'нет'}\n"
        f"Данные о клиенте: {profile_to_text(state.get('profile') or {})}\n"
        f"История диалога:\n{_history(messages)}\n\n"
        f"Новое сообщение клиента: {last}"
    )
    llm = get_llm("light").with_structured_output(Requirements)
    try:
        req = await llm.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)])
    except Exception as exc:
        req = None
        error = f"{type(exc).__name__}"
    else:
        error = "пустой ответ модели"

    if req is None:
        return {
            "analyzed": True,
            "intent": "other",
            "answer": f"Не удалось разобрать сообщение ({error}). Попробуйте переформулировать заявку.",
        }

    req.quantity = max(1, req.quantity or 1)
    update: dict = {"analyzed": True, "intent": req.intent}

    if req.intent == "order":
        update["requirements"] = req.model_dump()
        if req.category is None:
            update["answer"] = req.question or ASK_CATEGORY
    elif req.intent == "other":
        update["answer"] = OTHER_REPLY
    # для consult прежние требования остаются нетронутыми: контекст заказа не теряется
    return update
