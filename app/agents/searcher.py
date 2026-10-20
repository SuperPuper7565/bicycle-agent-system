"""Агент поиска по каталогу. Работает через инструменты СВОЕГО MCP-сервера."""
import json

from ..llm import get_llm
from ..logic import extract_products, hard_filter, search_args, text_of
from .tool_agent import run_tool_agent

SYSTEM = """Ты агент поиска по каталогу велосипедов. Единственный источник данных: инструмент search_catalog.
1. Вызови search_catalog с параметрами из требований клиента: category, а также max_price и
   wheel_size, если они заданы.
2. Если результатов нет, вызови search_catalog ещё раз только с category, чтобы собрать
   ближайшие альтернативы для ответа клиенту.
3. Не выдумывай товары и цены. В конце одним предложением скажи, что нашёл."""


def make_searcher(tools: list):
    by_name = {t.name: t for t in tools}
    search_tools = [t for t in tools if t.name in ("search_catalog", "get_product")]

    async def searcher_node(state: dict) -> dict:
        req = state.get("requirements") or {}
        task = "Требования клиента (JSON): " + json.dumps(req, ensure_ascii=False)
        messages = await run_tool_agent(get_llm("heavy"), search_tools, SYSTEM, task, max_steps=4)
        products, meta = extract_products(messages)

        # Страховка: модель не вызвала инструмент -> делаем обязательный вызов сами.
        if not products and "search_catalog" in by_name and not any(
            getattr(m, "name", None) == "search_catalog" for m in messages
        ):
            try:
                raw = text_of(await by_name["search_catalog"].ainvoke(search_args(req)))
                from types import SimpleNamespace

                products, meta = extract_products([SimpleNamespace(name="search_catalog", content=raw)])
            except Exception as exc:
                meta = {"source": "error", "note": f"Каталог недоступен: {type(exc).__name__}"}

        candidates, rejected = hard_filter(products, req)
        return {
            "products": products,
            "candidates": candidates,
            "rejected": rejected,
            "search_meta": meta,
            "searched": True,
        }

    return searcher_node
