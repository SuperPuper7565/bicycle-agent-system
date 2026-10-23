"""Чистая логика без LLM и без langchain: фильтры, маршрутизация, разбор ответов, шаблон КП.

Всё, что можно посчитать кодом, считается здесь: так числа в КП не зависят от
«настроения» модели, а логику можно проверить тестами.
"""
from __future__ import annotations

import json
import re
from datetime import date

CATEGORY_RU = {
    "mountain": "горный",
    "road": "шоссейный",
    "city": "городской",
    "hybrid": "гибридный",
    "kids": "детский",
    "teen": "подростковый",
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
    if req.get("child_age") or req.get("child_height_cm"):
        bits = []
        if req.get("child_age"):
            bits.append(f"{req['child_age']} лет")
        if req.get("child_height_cm"):
            bits.append(f"рост {req['child_height_cm']} см")
        parts.append("ребёнок: " + ", ".join(bits))
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
            qty = req.get("quantity") or 1
            lines.append(f"   - Количество: {qty}, итого: {format_rub(calc['total'])}")
        if p.get("url"):
            lines.append(f"   - Ссылка: {p['url']}")
    if texts.get("recommendation"):
        lines += ["", "## Рекомендация", texts["recommendation"]]
    if texts.get("advice"):
        lines += ["", "## На что обратить внимание", texts["advice"]]
    lines += ["", "## Примечание", "Цены, наличие, сроки поставки, условия оплаты и доставки подтверждаются менеджером."]
    if meta.get("source") == "cache":
        lines.append("Данные каталога получены из локального кэша, перед отправкой клиенту актуальность цен нужно проверить.")
    return "\n".join(lines)


# ---------- разбор заявки: контекст, выдумки и вопросы решает код, а не модель ----------

ASK_CATEGORY = "Какой тип велосипеда вам нужен: горный, шоссейный, городской, гибридный или детский?"
ASK_CHILD_AGAIN = "Возраст ребёнка указан необычно. Напишите, пожалуйста, сколько ему лет (до 17) или какой у него рост в см."
ASK_CHILD = "Сколько лет ребёнку или какой у него рост? От этого зависит размер колёс."

# По этим словам считаем, что клиент сам назвал тип велосипеда. Иначе значение модели отбрасывается.
CATEGORY_HINTS = {
    "mountain": r"\b(?:горн|mtb|мтб|маунтин|бездорож|грунт|лес)",
    "road": r"\b(?:шосс|асфальт)",
    "city": r"\b(?:город|дорожн|прогулоч|работ)",
    "hybrid": r"\bгибрид",
    "kids": r"\b(?:дет|ребен|сын|доч|внук|малыш|школьн)",
}

# Числа словами (текст сравнивается после замены ё на е).
NUM_WORDS = {
    "один": 1, "одного": 1, "одна": 1, "два": 2, "две": 2, "двух": 2, "три": 3, "трех": 3,
    "четыре": 4, "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15,
    "пятнадцати": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40, "сорока": 40,
    "пятьдесят": 50, "пятидесяти": 50, "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70,
    "семидесяти": 70, "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90,
    "сто": 100, "ста": 100,
}

PAST_ORDER_RE = re.compile(
    r"как\s+(?:в\s+)?прошл|как\s+раньше|как\s+обычно|повтори|такой\s+же\s+как|то\s+же\s+самое",
    re.IGNORECASE,
)

SCALAR_FIELDS = ("category", "quantity", "max_price", "wheel_size", "child_height_cm", "child_age")


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е").replace("\xa0", " ")


def _as_float(value):
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def numbers_in(text: str) -> set[float]:
    """Все числа из сообщения: цифрами, «20 тысяч», «20к», «двадцать пять тысяч», «пять»."""
    t = _norm(text)
    nums: set[float] = set()
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(?:тысяч\w*|тыс\b\.?|к\b)", t):
        nums.add(float(m.group(1).replace(",", ".")) * 1000)
    for m in re.finditer(r"\d{1,3}(?:\s\d{3})+|\d+(?:[.,]\d+)?", t):
        nums.add(float(re.sub(r"\s", "", m.group(0)).replace(",", ".")))
    for m in re.finditer(r"((?:[а-я]+\s+){0,2}[а-я]+)\s+тысяч\w*", t):
        values = [NUM_WORDS[w] for w in m.group(1).split() if w in NUM_WORDS]
        if values:
            nums.add(float(sum(values) * 1000))
    for w in re.findall(r"[а-я]+", t):
        if w in NUM_WORDS:
            nums.add(float(NUM_WORDS[w]))
    return nums


def category_supported(category: str | None, text: str) -> bool:
    pattern = CATEGORY_HINTS.get(category or "")
    return bool(pattern and re.search(pattern, _norm(text)))


def _phrase_supported(phrase: str, text_norm: str) -> bool:
    words = re.findall(r"[a-zа-я0-9]{4,}", _norm(phrase))
    return not words or any(w[:5] in text_norm for w in words)


def ground_requirements(new: dict, text: str) -> dict:
    """Убрать значения, которых клиент не называл (защита от выдумок слабой модели)."""
    out = dict(new)
    nums = numbers_in(text)
    text_norm = _norm(text)
    qty = out.get("quantity") or 1

    price = _as_float(out.get("max_price"))
    if out.get("max_price") is not None and (price is None or not ({price, price * qty} & nums)):
        out["max_price"] = None  # допускаем и общий бюджет на несколько штук
    for key in ("wheel_size", "child_age", "child_height_cm"):
        value = _as_float(out.get(key))
        if out.get(key) is not None and (value is None or value not in nums):
            out[key] = None
    age, height = _as_float(out.get("child_age")), _as_float(out.get("child_height_cm"))
    if age is not None and not 1 <= age <= 17:
        out["child_age"] = None  # «ребёнку 67 лет»: неправдоподобно, спросим ещё раз
        out["bad_age"] = True
    if height is not None and not 60 <= height <= 200:
        out["child_height_cm"] = None
    if out.get("category") and not category_supported(out["category"], text):
        out["category"] = None
    if out.get("for_child") and not category_supported("kids", text):
        out["for_child"] = False
    for key in ("must_have", "wishes"):
        out[key] = [p for p in (out.get(key) or []) if _phrase_supported(p, text_norm)]
    return out


SEARCH_RE = re.compile(r"найд|ищи|иска|поищ|подбер|покаж|давай|начин", re.IGNORECASE)


def wants_search_now(text: str) -> bool:
    """Клиент просит выполнить поиск («найди уже на сайте», «а ты будешь искать?»)."""
    return bool(SEARCH_RE.search(_norm(text)))


def has_order_data(req: dict) -> bool:
    """В сообщении есть данные заказа: тип, ребёнок или параметры."""
    return bool(
        req.get("category") or req.get("for_child") or req.get("bad_age")
        or any(req.get(k) for k in ("max_price", "wheel_size", "child_age", "child_height_cm", "quantity"))
        or req.get("must_have") or req.get("wishes")
    )


def wants_past_order(text: str) -> bool:
    return bool(PAST_ORDER_RE.search(text or ""))


def merge_requirements(prev: dict | None, new: dict, *, awaiting: bool, text: str, profile: dict | None):
    """Собрать итоговые требования. Возвращает (требования, пояснение, продолжена_ли_прошлая_заявка).

    Прошлая заявка переносится только если ассистент только что задал вопрос (awaiting) или клиент
    уточняет её («а подешевле»). Иначе это новая заявка с чистого листа. Профиль клиента
    используется только по прямой ссылке («как в прошлый раз»).
    """
    prev = prev or {}
    has_params = (
        any(new.get(k) for k in ("max_price", "wheel_size", "child_age", "child_height_cm", "quantity"))
        or bool(new.get("must_have"))
        or bool(new.get("wishes"))
    )
    refines = bool(new.get("refines_previous")) or (
        not new.get("category") and not new.get("for_child") and has_params and bool(prev.get("category"))
    )
    carried = bool(prev) and (awaiting or refines)
    if (carried and awaiting and new.get("category") and prev.get("category")
            and new["category"] != prev["category"]
            and not (prev["category"] in ("kids", "teen") and (prev.get("child_age") or prev.get("child_height_cm")))):
        carried = False  # на вопрос не ответили и назвали другой тип: это новая заявка

    base = dict(prev) if carried else {}
    merged: dict = dict(base)
    for key in SCALAR_FIELDS:
        if new.get(key) is not None:
            merged[key] = new[key]
    for key in ("must_have", "wishes"):
        merged[key] = list(dict.fromkeys((base.get(key) or []) + (new.get(key) or [])))
    merged["for_child"] = bool(base.get("for_child") or new.get("for_child"))

    note = "уточнение прошлой заявки" if carried else "новая заявка"
    if wants_past_order(text) and profile:
        filled = False
        for pkey, field in (("last_category", "category"), ("last_max_price", "max_price"),
                            ("last_wheel_size", "wheel_size"), ("last_quantity", "quantity")):
            if merged.get(field) is None and profile.get(pkey):
                merged[field] = profile[pkey]
                filled = True
        if filled:
            note += ", дополнено из профиля клиента"
    if merged["for_child"] and not merged.get("category"):
        merged["category"] = "kids"
    if merged.get("category") == "kids":
        age, height = _as_float(merged.get("child_age")), _as_float(merged.get("child_height_cm"))
        if (age is not None and age >= 14) or (height is not None and height >= 150):
            merged["category"] = "teen"  # подросток: в детском разделе подходящих размеров нет
    merged["bad_age"] = bool(new.get("bad_age"))
    return merged, note, carried


def next_question(req: dict, child_asked: bool) -> str | None:
    """Уточняющий вопрос нужен только в двух случаях: неизвестен тип или для ребёнка неизвестен размер."""
    if req.get("bad_age") and not (req.get("child_height_cm") or req.get("wheel_size")):
        return ASK_CHILD_AGAIN
    if not req.get("category"):
        return ASK_CATEGORY
    has_size_info = req.get("child_age") or req.get("child_height_cm") or req.get("wheel_size")
    if req["category"] == "kids" and not has_size_info and not child_asked:
        return ASK_CHILD
    return None
