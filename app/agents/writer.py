"""Агент-составитель КП. Числа считает MCP-инструмент calculate_total, текст пишет LLM."""
import json
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..logic import no_results_text, parse_tool_json, render_offer, text_of
from ..schemas import OfferText

SYSTEM = """Ты составитель коммерческих предложений магазина велосипедов. Напиши три коротких текста.
- intro: 1-2 вежливых предложения вступления.
- recommendation: 2-3 предложения, какой из вариантов лучше подходит и почему.
  Опирайся только на переданные данные.
- conditions: условия доставки, сборки, оплаты и гарантии из блока «Условия из базы знаний».
  Если блок пуст, напиши, что условия уточнит менеджер.
Не выдумывай характеристики, цены, скидки и сроки. Цены и расчёты в тексте не повторяй."""

TOP_N = 3


def make_writer(tools: list):
    by_name = {t.name: t for t in tools}

    async def _calc(item: dict, quantity: int):
        tool = by_name.get("calculate_total")
        if tool is None:
            return None
        try:
            raw = text_of(await tool.ainvoke(
                {"items": [{"name": item.get("name"), "price": item.get("price")}], "quantity": quantity}
            ))
        except Exception:
            return None
        data = parse_tool_json(raw)
        if isinstance(data, dict) and "total" not in data and isinstance(data.get("result"), dict):
            data = data["result"]
        return data if isinstance(data, dict) and "total" in data else None

    async def writer_node(state: dict) -> dict:
        req = state.get("requirements") or {}
        meta = state.get("search_meta") or {}
        ranked = state.get("ranked") or []

        if not ranked:
            return {"answer": no_results_text(req, state.get("rejected") or [], meta), "written": True}

        quantity = req.get("quantity") or 1
        top = ranked[:TOP_N]
        rows = [{"product": p, "calc": await _calc(p, quantity)} for p in top]

        facts = [
            {"name": p.get("name"), "price": p.get("price"), "specs": p.get("specs"),
             "score": (p.get("check") or {}).get("score"), "comment": (p.get("check") or {}).get("comment")}
            for p in top
        ]
        prompt = (
            f"Заявка: {json.dumps(req, ensure_ascii=False)}\n"
            f"Варианты (от лучшего):\n{json.dumps(facts, ensure_ascii=False, indent=1)}\n\n"
            f"Условия из базы знаний:\n{state.get('kb_notes') or '(нет данных)'}"
        )
        texts: dict = {}
        try:
            result = await get_llm("heavy", temperature=0.2).with_structured_output(OfferText).ainvoke(
                [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
            )
            if result is not None:
                texts = result.model_dump()
        except Exception:
            texts = {}
        if not texts.get("conditions") and state.get("kb_notes"):
            texts["conditions"] = state["kb_notes"]  # условия из RAG не теряем даже при сбое модели

        offer = render_offer(req, rows, texts, date.today(), meta)
        return {"answer": offer, "written": True}

    return writer_node
