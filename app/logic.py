"""Чистая логика без LLM и без langchain: фильтры, маршрутизация, разбор ответов, шаблон КП.

Всё, что можно посчитать кодом, считается здесь: так числа в КП не зависят от
«настроения» модели, а логику можно проверить тестами.
"""
from __future__ import annotations

import json
from datetime import date

CATEGORY_RU = {
    "mountain": "горный",
    "road": "шоссейный",
    "city": "городской",
    "hybrid": "гибридный",
    "kids": "детский",
}

SPEC_LABELS = {
    "wheel_size": "колёса {}″",
    "frame": "рама: {}",
    "speeds": "скоростей: {}",
    "brakes": "тормоза: {}",
}

MAX_STEPS = 12


# ---------- вспомогательное ----------

def text_of(content) -> str:
    """Привести content сообщения/инструмента (строка или список блоков) к строке."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or "")
            else:
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts)
    return str(content)


def parse_tool_json(text: str):
    """Достать JSON из ответа инструмента (устойчиво к лишнему тексту вокруг)."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    if not isinstance(text, str):
        return None
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            return None
    return None


def format_rub(n) -> str:
    return f"{int(n):,}".replace(",", " ") + " ₽"


def normalize_wheel(value) -> str:
    s = str(value).replace(",", ".").replace('"', "").replace("″", "").strip()
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def format_specs(specs: dict | None) -> str:
    if not specs:
        return ""
    parts = []
    for key, value in specs.items():
        template = SPEC_LABELS.get(key)
        parts.append(template.format(value) if template else f"{key}: {value}")
    return ", ".join(parts)


def summarize_requirements(req: dict) -> str:
    parts = []
    if req.get("category"):
        parts.append(f"тип: {CATEGORY_RU.get(req['category'], req['category'])}")
    parts.append(f"количество: {req.get('quantity') or 1}")
    if req.get("max_price"):
        parts.append(f"бюджет: до {format_rub(req['max_price'])} за единицу")
    if req.get("wheel_size"):
        parts.append(f"колёса: {normalize_wheel(req['wheel_size'])}″")
    if req.get("must_have"):
        parts.append("обязательно: " + ", ".join(req["must_have"]))
    if req.get("wishes"):
        parts.append("пожелания: " + ", ".join(req["wishes"]))
    return "; ".join(parts)


# ---------- каталог ----------

def search_args(req: dict) -> dict:
    args: dict = {"category": req.get("category")}
    if req.get("max_price"):
        args["max_price"] = int(req["max_price"])
    if req.get("wheel_size"):
        args["wheel_size"] = normalize_wheel(req["wheel_size"])
    return args


def extract_products(messages) -> tuple[list[dict], dict]:
    """Собрать товары из всех ответов инструмента search_catalog (без дублей)."""
    products: list[dict] = []
    seen: set = set()
    meta: dict = {}
    for m in messages:
        if getattr(m, "name", None) != "search_catalog":
            continue
        data = parse_tool_json(text_of(getattr(m, "content", None)))
        if isinstance(data, dict) and "items" not in data and isinstance(data.get("result"), (dict, list)):
            data = data["result"]  # некоторые версии MCP оборачивают ответ в {"result": ...}
        if isinstance(data, dict):
            items = data.get("items") or []
            if data.get("source"):
                meta = {"source": data.get("source"), "note": data.get("note", "")}
        elif isinstance(data, list):
            items = data
        else:
            continue
        for p in items:
            key = p.get("url") or p.get("name")
            if key in seen:
                continue
            seen.add(key)
            products.append(p)
    return products, meta


def hard_filter(products: list[dict], req: dict) -> tuple[list[dict], list[dict]]:
    """Жёсткие требования проверяются кодом: категория, бюджет, диаметр колёс."""
    ok: list[dict] = []
    rejected: list[dict] = []
    category = req.get("category")
    max_price = req.get("max_price")
    wheel = req.get("wheel_size")
    for p in products:
        reasons: list[str] = []
        if category and p.get("category") and p["category"] != category:
            reasons.append("другая категория")
        price = p.get("price")
        if price is None:
            reasons.append("цена не указана")
        elif max_price and price > max_price:
            reasons.append(f"дороже бюджета на {format_rub(price - max_price)}")
        product_wheel = (p.get("specs") or {}).get("wheel_size")
        if wheel and product_wheel and normalize_wheel(product_wheel) != normalize_wheel(wheel):
            reasons.append(f"колёса {normalize_wheel(product_wheel)}″ вместо {normalize_wheel(wheel)}″")
        if reasons:
            rejected.append({"product": p, "reasons": reasons})
        else:
            ok.append(p)
    return ok, rejected


def rank_candidates(candidates: list[dict], checks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Объединить кандидатов с отчётом проверщика. Вернуть (подходящие по убыванию оценки, отсеянные)."""
    by_url = {c.get("url"): c for c in checks}
    ranked: list[dict] = []
    dropped: list[dict] = []
    for p in candidates:
        check = by_url.get(p.get("url")) or {"suitable": True, "met": [], "unmet": [], "comment": "", "score": 5}
        if not check.get("suitable", True):
            reason = "не выполнено обязательное: " + (", ".join(check.get("unmet") or []) or "требование")
            dropped.append({"product": p, "reasons": [reason]})
            continue
        ranked.append({**p, "check": check})
    ranked.sort(key=lambda x: (-int(x["check"].get("score", 5)), x.get("price") or 0))
    return ranked, dropped


# ---------- маршрутизация ----------

def allowed_routes(state: dict) -> list[str]:
    """Какие шаги допустимы сейчас. Supervisor выбирает только из этого списка."""
    if state.get("answer") or state.get("written"):
        return ["END"]
    if not state.get("analyzed"):
        return ["analyst"]
    intent = state.get("intent")
    if intent == "consult":
        return ["END"] if state.get("advised") else ["advisor"]
    if intent != "order":
        return ["END"]
    if not state.get("searched"):
        return ["searcher"]
    if not state.get("checked"):
        return ["checker"] if state.get("candidates") else ["writer"]
    if state.get("ranked") and not state.get("advised"):
        return ["advisor", "writer"]
    return ["writer"]


# ---------- тексты ----------

def no_results_text(req: dict, rejected: list[dict], meta: dict) -> str:
    lines = [
        "К сожалению, по вашей заявке подходящих вариантов не нашлось.",
        "",
        "**Ваши требования:** " + summarize_requirements(req),
    ]
    if meta.get("source") == "cache":
        lines.append("_Данные получены из локального кэша каталога, актуальность нужно проверить._")
    if rejected:
        nearest = sorted(rejected, key=lambda r: r["product"].get("price") or 10**9)[:3]
        lines += ["", "**Ближайшие варианты, которые не подошли:**"]
        for r in nearest:
            p = r["product"]
            price = format_rub(p["price"]) if p.get("price") is not None else "цена не указана"
            lines.append(f"- {p.get('name')} — {price}: {'; '.join(r['reasons'])}")
    tips: list[str] = []
    budget_only = [
        r for r in rejected
        if len(r["reasons"]) == 1 and r["reasons"][0].startswith("дороже бюджета")
    ]
    if budget_only:
        cheapest = min(budget_only, key=lambda r: r["product"]["price"])["product"]
        tips.append(f"увеличить бюджет до {format_rub(cheapest['price'])} (подойдёт «{cheapest.get('name')}»)")
    if any(reason.startswith("колёса") for r in rejected for reason in r["reasons"]):
        tips.append("рассмотреть другой диаметр колёс")
    if req.get("must_have"):
        tips.append("смягчить обязательные требования")
    if tips:
        lines += ["", "**Что можно изменить:** " + "; ".join(tips) + "."]
    return "\n".join(lines)


def render_offer(req: dict, rows: list[dict], texts: dict, today: date, meta: dict) -> str:
    """Собрать КП. Цены, скидки и итоги берутся из данных, а не из текста модели."""
    lines = [f"# Коммерческое предложение от {today:%d.%m.%Y}", ""]
    if texts.get("intro"):
        lines += [texts["intro"], ""]
    lines += ["## Параметры заявки", summarize_requirements(req), "", "## Подходящие варианты"]
    for i, row in enumerate(rows, 1):
        p = row["product"]
        calc = row.get("calc") or {}
        lines.append(f"{i}. **{p.get('name')}**: {format_rub(p['price'])} за единицу")
        specs = format_specs(p.get("specs"))
        if specs:
            lines.append(f"   - Характеристики: {specs}")
        check = p.get("check") or {}
        if check.get("met"):
            lines.append("   - Соответствует: " + ", ".join(check["met"]))
        if check.get("unmet"):
            lines.append("   - Не выполнено: " + ", ".join(check["unmet"]))
        if calc.get("total") is not None:
            qty = (req.get("quantity") or 1)
            discount = calc.get("discount_percent", 0)
            extra = f", скидка {discount}%" if discount else ""
            lines.append(f"   - Количество: {qty}{extra}, итого: {format_rub(calc['total'])}")
        if p.get("url"):
            lines.append(f"   - Ссылка: {p['url']}")
    if texts.get("recommendation"):
        lines += ["", "## Рекомендация", texts["recommendation"]]
    if texts.get("conditions"):
        lines += ["", "## Условия", texts["conditions"]]
    lines += ["", "## Примечание", "Наличие и срок поставки подтверждаются менеджером. Цены действительны 3 рабочих дня."]
    if meta.get("source") == "cache":
        lines.append("Данные каталога получены из локального кэша, перед отправкой клиенту актуальность цен нужно проверить.")
    return "\n".join(lines)
