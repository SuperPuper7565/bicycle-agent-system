"""Агент-консультант: отвечает на вопросы и достаёт условия из базы знаний (RAG)."""
from ..llm import get_llm
from ..logic import text_of
from ..rag.retriever import kb_search
from .tool_agent import final_text, run_tool_agent

SYSTEM = """Ты консультант магазина велосипедов. Отвечай ТОЛЬКО на основе результатов инструмента kb_search
(база знаний магазина). Сделай 1-3 поисковых запроса, затем дай краткий точный ответ на русском.
Если в базе знаний ответа нет, так и скажи, ничего не выдумывай. Цен и наличия в базе знаний нет."""


async def advisor_node(state: dict) -> dict:
    intent = state.get("intent")
    req = state.get("requirements") or {}
    if intent == "consult":
        task = "Вопрос клиента: " + text_of(state["messages"][-1].content)
    else:
        qty = req.get("quantity") or 1
        task = (
            f"Найди условия для заказа {qty} шт.: доставка, сборка, оплата, гарантия, "
            "оптовые скидки. Изложи кратко, списком фактов."
        )

    messages = await run_tool_agent(get_llm("heavy"), [kb_search], SYSTEM, task, max_steps=4)
    text = final_text(messages)
    update: dict = {"advised": True, "kb_notes": text}
    if intent == "consult":
        update["answer"] = text or "В базе знаний нет ответа на этот вопрос. Уточните у менеджера."
    return update
