"""Минимальный ReAct-цикл: модель вызывает инструменты, пока не выдаст итоговый текст.

Нужен агентам, которые работают с инструментами (поиск по каталогу через MCP,
поиск по базе знаний через RAG). Написан вручную, чтобы не зависеть от версии
готовых хелперов (create_react_agent и т.п.).
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..logic import text_of


async def run_tool_agent(llm, tools, system: str, task: str, max_steps: int = 4) -> list:
    """Вернуть полный список сообщений (включая ToolMessage с результатами)."""
    by_name = {t.name: t for t in tools}
    bound = llm.bind_tools(tools) if tools else llm
    messages: list = [SystemMessage(content=system), HumanMessage(content=task)]

    for _ in range(max_steps):
        ai = await bound.ainvoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            break
        for call in calls:
            tool = by_name.get(call["name"])
            if tool is None:
                result = f"Неизвестный инструмент: {call['name']}"
            else:
                try:
                    result = text_of(await tool.ainvoke(call["args"]))
                except Exception as exc:  # ошибка инструмента не должна ронять граф
                    result = f"Ошибка инструмента {call['name']}: {type(exc).__name__}: {exc}"
            messages.append(ToolMessage(content=result, tool_call_id=call["id"], name=call["name"]))
    return messages


def final_text(messages: list) -> str:
    """Текст последнего ответа модели без вызовов инструментов."""
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai" and not getattr(m, "tool_calls", None):
            return text_of(m.content).strip()
    return ""
