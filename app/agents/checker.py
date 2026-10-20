"""Агент-проверщик (тяжёлая модель): сверяет товары с обязательными требованиями и пожеланиями.

Жёсткие требования (тип, бюджет, колёса) уже проверены кодом в searcher. Здесь LLM
оценивает то, что кодом не проверить: тормоза, рама, «недорогой», «лёгкий» и т.п.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..logic import rank_candidates
from ..schemas import CheckReport

SYSTEM = """Ты проверщик соответствия товаров требованиям клиента. Для КАЖДОГО товара из списка верни проверку.
- suitable=false только если не выполнено ОБЯЗАТЕЛЬНОЕ требование (must_have). Невыполненные
  пожелания (wishes) лишь снижают score.
- Опирайся только на данные товара (название, цена, характеристики). Если характеристики нет,
  не предполагай, что она есть: напиши в unmet «нет данных: ...» и снизь score.
- «Недорогой» оценивай относительно других товаров в списке.
- met и unmet: короткие фразы по требованиям клиента. score: 0-10.
- url копируй дословно из данных товара."""


async def checker_node(state: dict) -> dict:
    req = state.get("requirements") or {}
    candidates = (state.get("candidates") or [])[:8]
    checks: list[dict] = []

    if req.get("must_have") or req.get("wishes"):
        slim = [
            {"url": p.get("url"), "name": p.get("name"), "price": p.get("price"), "specs": p.get("specs")}
            for p in candidates
        ]
        prompt = (
            f"Обязательные требования (must_have): {json.dumps(req.get('must_have') or [], ensure_ascii=False)}\n"
            f"Пожелания (wishes): {json.dumps(req.get('wishes') or [], ensure_ascii=False)}\n"
            f"Товары:\n{json.dumps(slim, ensure_ascii=False, indent=1)}"
        )
        llm = get_llm("heavy").with_structured_output(CheckReport)
        try:
            report = await llm.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=prompt)])
            if report is not None:
                valid = {p.get("url") for p in candidates}
                checks = [c.model_dump() for c in report.checks if c.url in valid]
        except Exception:
            checks = []  # без отчёта товары остаются с нейтральной оценкой

    ranked, dropped = rank_candidates(candidates, checks)
    return {
        "checks": checks,
        "ranked": ranked,
        "rejected": (state.get("rejected") or []) + dropped,
        "checked": True,
    }
