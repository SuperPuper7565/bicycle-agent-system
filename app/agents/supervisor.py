"""Supervisor (оркестратор): решает, какой агент работает следующим.

Допустимые шаги вычисляет код (logic.allowed_routes), так что граф не зациклится и не
перескочит этап. Когда допустим один вариант, LLM не вызывается. Когда есть выбор
(например, запросить условия у консультанта перед КП), решает модель.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..logic import MAX_STEPS, allowed_routes, summarize_requirements
from ..schemas import Route

SYSTEM = """Ты supervisor (оркестратор) команды агентов магазина велосипедов.
Агенты: analyst (разбор заявки), searcher (поиск в каталоге через MCP), checker (проверка требований),
advisor (база знаний: справка по характеристикам, размерам и выбору), writer (коммерческое предложение).
Выбери СЛЕДУЮЩЕГО исполнителя строго из списка допустимых.
Правило: в коммерческое предложение полезно включить справку по выбору, поэтому если advisor ещё не
вызывался, выбирай advisor, а затем writer."""


def _describe(state: dict) -> str:
    req = state.get("requirements") or {}
    return (
        f"Заявка: {summarize_requirements(req) if req else 'нет'}\n"
        f"Найдено товаров: {len(state.get('products') or [])}; подходящих после проверки: {len(state.get('ranked') or [])}\n"
        f"Справка из базы знаний получена: {'да' if state.get('advised') else 'нет'}"
    )


async def supervisor_node(state: dict) -> dict:
    steps = int(state.get("steps", 0)) + 1
    allowed = allowed_routes(state)

    if steps > MAX_STEPS:
        return {"route": "END", "route_reason": "превышен лимит шагов", "steps": steps}
    if len(allowed) == 1:
        return {"route": allowed[0], "route_reason": "единственный допустимый шаг", "steps": steps}

    prompt = f"{_describe(state)}\nДопустимые исполнители: {', '.join(allowed)}"
    try:
        decision = await get_llm("heavy").with_structured_output(Route).ainvoke(
            [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
        )
    except Exception:
        decision = None

    if decision is not None and decision.next in allowed:
        return {"route": decision.next, "route_reason": decision.reason, "steps": steps}
    return {"route": allowed[0], "route_reason": "решение модели недопустимо, взят первый допустимый шаг", "steps": steps}
