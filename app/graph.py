"""Граф LangGraph: supervisor оркестрирует агентов, каждый агент возвращает управление supervisor.

            ┌────────────────────────────────────────────┐
  START → start_turn → supervisor ⇄ analyst / searcher / checker / advisor / writer
                           └─ END → finalize → END
"""
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from .agents.advisor import advisor_node
from .agents.analyst import analyst_node
from .agents.checker import checker_node
from .agents.searcher import make_searcher
from .agents.supervisor import supervisor_node
from .agents.writer import make_writer
from .memory import load_profile, update_profile
from .state import State

WORKERS = ["analyst", "searcher", "checker", "advisor", "writer"]


def start_turn(state: dict) -> dict:
    """Новая реплика клиента: сбрасываем результаты прошлой реплики, подгружаем профиль.

    requirements и messages сохраняются: так работает краткосрочная память диалога.
    """
    return {
        "profile": load_profile(state.get("user_id") or "default"),
        "intent": "",
        "products": [], "candidates": [], "rejected": [], "checks": [], "ranked": [],
        "kb_notes": "", "search_meta": {}, "answer": "",
        "analyzed": False, "searched": False, "checked": False, "advised": False, "written": False,
        "steps": 0, "route": "", "route_reason": "",
    }


def finalize(state: dict) -> dict:
    """Записываем ответ в историю диалога и сохраняем предпочтения клиента (долговременная память)."""
    answer = state.get("answer") or "Не удалось сформировать ответ. Попробуйте переформулировать запрос."
    if state.get("intent") == "order" and state.get("written") and state.get("ranked"):
        update_profile(state.get("user_id") or "default", state.get("requirements") or {})
    return {"messages": [AIMessage(content=answer)], "answer": answer}


def build_graph(tools: list, checkpointer=None):
    g = StateGraph(State)
    g.add_node("start_turn", start_turn)
    g.add_node("supervisor", supervisor_node)
    g.add_node("analyst", analyst_node)
    g.add_node("searcher", make_searcher(tools))
    g.add_node("checker", checker_node)
    g.add_node("advisor", advisor_node)
    g.add_node("writer", make_writer(tools))
    g.add_node("finalize", finalize)

    g.add_edge(START, "start_turn")
    g.add_edge("start_turn", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["route"],
        {**{w: w for w in WORKERS}, "END": "finalize"},
    )
    for worker in WORKERS:
        g.add_edge(worker, "supervisor")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
