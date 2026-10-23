"""Консольный чат с системой.

  python -m app.main                       # новый диалог при каждом запуске (клиент default)
  python -m app.main --user ivan --thread t1
  python -m app.main --graph               # вывести схему графа в формате Mermaid

Команды в чате: /new (новый диалог, профиль клиента сохраняется), /profile, /exit
"""
import argparse
import asyncio
import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from . import config
from .graph import build_graph
from .mcp_client import load_mcp_tools
from .logic import summarize_requirements
from .memory import load_profile, profile_to_text


def _print_update(node: str, data: dict | None) -> None:
    data = data or {}
    if node == "supervisor":
        print(f"  [supervisor] -> {data.get('route')}  ({data.get('route_reason')})")
    elif node == "analyst":
        req = data.get("requirements")
        if req:
            print(f"  [analyst] {data.get('req_note', '')}: {summarize_requirements(req)}")
        else:
            print(f"  [analyst] тип запроса: {data.get('intent')}")
    elif node not in ("start_turn", "finalize"):
        print(f"  [{node}] готово")


async def chat(user_id: str, thread_id: str) -> None:
    print("Загружаю инструменты MCP-сервера...")
    tools = await load_mcp_tools()
    print("Инструменты:", ", ".join(t.name for t in tools))

    config.CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as saver:
        graph = build_graph(tools, saver)
        print(f"\nКлиент: {user_id}; диалог: {thread_id}. Команды: /new, /profile, /exit")
        print("Память клиента:", profile_to_text(load_profile(user_id)))

        while True:
            try:
                text = input("\nВы: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text == "/exit":
                break
            if text == "/new":
                thread_id = uuid.uuid4().hex[:8]
                print(f"Новый диалог: {thread_id}")
                continue
            if text == "/profile":
                print(profile_to_text(load_profile(user_id)))
                continue

            cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
            payload = {"messages": [HumanMessage(content=text)], "user_id": user_id}
            try:
                async for chunk in graph.astream(payload, cfg, stream_mode="updates"):
                    for node, data in chunk.items():
                        _print_update(node, data)
                state = await graph.aget_state(cfg)
                print("\nАссистент:\n" + state.values["messages"][-1].content)
            except Exception as exc:
                print(f"\nОшибка: {type(exc).__name__}: {exc}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="default")
    parser.add_argument("--thread", default=None, help="id диалога (по умолчанию новый при каждом запуске)")
    parser.add_argument("--graph", action="store_true", help="вывести схему графа (Mermaid)")
    args = parser.parse_args()

    if args.graph:
        print(build_graph([]).get_graph().draw_mermaid())
        return
    asyncio.run(chat(args.user, args.thread or uuid.uuid4().hex[:8]))


if __name__ == "__main__":
    main()
