"""Агент-консультант: отвечает на вопросы и готовит справку по базе знаний (RAG)."""
import json

from ..llm import get_llm
from ..logic import text_of
from ..rag.retriever import kb_search
from .tool_agent import final_text, run_tool_agent

SYSTEM = """Ты консультант магазина велосипедов. Отвечай ТОЛЬКО на основе результатов инструмента kb_search
(база знаний с общими сведениями: типы велосипедов, подбор размера рамы и колёс, глоссарий характеристик, FAQ).
Сделай 1-3 поисковых запроса, затем дай краткий точный ответ на русском.
Если в базе знаний ответа нет, так и скажи, ничего не выдумывай.
В базе знаний НЕТ сведений о ценах, наличии, доставке, оплате, гарантии, возврате, скидках и сроках:
на такие вопросы отвечай, что эти условия уточняет менеджер."""


async def advisor_node(state: dict) -> dict:
    intent = state.get("intent")
    req = state.get("requirements") or {}
    if intent == "consult":
        task = "Вопрос клиента: " + text_of(state["messages"][-1].content)
    else:
        facts = [
            {"name": p.get("name"), "specs": p.get("specs")}
            for p in (state.get("ranked") or [])[:3]
        ]
        task = (
            "Подготовь справку для коммерческого предложения: что означают характеристики найденных "
            "вариантов (диаметр колёс, рама, тормоза, подвеска, если они указаны) и на что обратить "
            "внимание при выборе (например, посадка и размер). Только общие сведения из базы знаний, "
            "3-5 коротких пунктов.\n"
            f"Требования клиента: {json.dumps(req, ensure_ascii=False)}\n"
            f"Найденные варианты: {json.dumps(facts, ensure_ascii=False)}"
        )

    messages = await run_tool_agent(get_llm("heavy"), [kb_search], SYSTEM, task, max_steps=4)
    text = final_text(messages)
    update: dict = {"advised": True, "kb_notes": text}
    if intent == "consult":
        update["answer"] = text or "В базе знаний нет ответа на этот вопрос. Уточните у менеджера."
    return update
