"""Агент-аналитик (лёгкая модель, Qwen): текст заявки -> структурированные требования.

Модель только ИЗВЛЕКАЕТ данные из последнего сообщения клиента. Что переносить из прошлых
сообщений, когда задавать вопрос и какие значения считать выдуманными, решает код (logic.py):
7B-модель плохо отличает «новая заявка» от «уточнение» и склонна дописывать детали из контекста.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..logic import (
    ASK_CHILD,
    ground_requirements,
    has_order_data,
    merge_requirements,
    next_question,
    text_of,
    wants_search_now,
)
from ..schemas import Requirements

SYSTEM = """Ты аналитик заявок магазина велосипедов. Извлеки данные ТОЛЬКО из сообщения клиента.
Ничего не додумывай: чего клиент не сказал, оставляй пустым (null, false, []).
- intent: "order": клиент хочет подобрать или купить велосипед (в том числе отвечает на вопрос
  ассистента о заказе); "consult": вопрос о выборе, размерах, характеристиках, доставке, оплате,
  гарантии без заказа; "other": приветствие или не по теме.
- category: mountain (горный), road (шоссейный), city (городской), hybrid (гибридный), kids (детский).
  Только если тип назван или однозначно следует из слов клиента. Иначе null.
- for_child: true, если велосипед для ребёнка (сын, дочь, внук, ребёнок, детский).
- child_age: возраст ребёнка в годах; child_height_cm: рост ребёнка в см. Только если названы.
- quantity: сколько велосипедов, только если клиент назвал число.
- max_price: максимум за ОДИН велосипед в рублях, числом, только если назван. «до 20 тысяч» = 20000.
  Если назван общий бюджет на несколько штук, раздели его на количество.
- wheel_size: диаметр колёс в дюймах строкой, например "29", только если назван.
- must_have: обязательные требования клиента («обязательно дисковые тормоза»).
- wishes: необязательные пожелания («недорогой», «лёгкий»), только сказанные клиентом.
- refines_previous: true, если сообщение меняет или дополняет прошлую заявку («а подешевле»,
  «а с дисковыми тормозами», «до 25 тысяч»), а не начинает новую."""

OTHER_REPLY = (
    "Я помогаю подобрать велосипед и подготовить коммерческое предложение. "
    "Опишите, что вам нужно: тип велосипеда, бюджет, диаметр колёс, количество. "
    "Также могу ответить на вопросы о типах велосипедов, подборе размера и характеристиках."
)


def _last_assistant_text(messages: list) -> str:
    for m in reversed(messages[:-1]):  # последнее сообщение это ответ клиента
        if getattr(m, "type", "") == "ai":
            return text_of(m.content)[:400]
    return ""


async def analyst_node(state: dict) -> dict:
    messages = state.get("messages", [])
    last = text_of(messages[-1].content) if messages else ""
    awaiting = bool(state.get("awaiting"))  # прошлая реплика ассистента была вопросом

    prompt = f"Сообщение клиента: {last}"
    if awaiting:
        question = _last_assistant_text(messages)
        if question:
            prompt = f"Вопрос ассистента: {question}\nОтвет клиента на этот вопрос: {last}"

    llm = get_llm("light").with_structured_output(Requirements)
    try:
        req = await llm.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)])
        error = "пустой ответ модели"
    except Exception as exc:
        req, error = None, type(exc).__name__

    if req is None:
        return {
            "analyzed": True,
            "intent": "other",
            "answer": f"Не удалось разобрать сообщение ({error}). Попробуйте переформулировать заявку.",
        }

    new = ground_requirements(req.model_dump(), last)  # убираем то, чего клиент не говорил
    intent = req.intent
    # Пока заявка открыта (мы задали вопрос), ответ клиента почти всегда относится к ней. 7B-модель
    # часто помечает такие ответы как consult, и поиск не запускается. Поэтому решает код:
    # заявка продолжается, если в ответе есть данные заказа или просьба искать; вопрос со знаком «?»
    # остаётся консультацией.
    asks = "?" in last and not wants_search_now(last)
    if awaiting and intent != "order" and not asks and (
        intent == "other" or has_order_data(new) or wants_search_now(last)
    ):
        intent = "order"
    update: dict = {"analyzed": True, "intent": intent}

    if intent == "order":
        merged, note, carried = merge_requirements(
            state.get("requirements"), new, awaiting=awaiting, text=last, profile=state.get("profile") or {}
        )
        child_asked = bool(state.get("child_asked")) and carried
        question = next_question(merged, child_asked)
        update.update(
            requirements=merged,
            req_note=note,
            awaiting=question is not None,
            child_asked=child_asked or question == ASK_CHILD,
        )
        if question:
            update["answer"] = question
    elif intent == "other":
        update["answer"] = OTHER_REPLY
    # consult: требования и флаг awaiting не трогаем: после справки заявку можно продолжить
    return update
