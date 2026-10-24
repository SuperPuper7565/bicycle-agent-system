"""Логика каталога: загрузка страниц velosklad.ru, детерминированный парсинг (без LLM).

Источники данных:
  live         страницы раздела с листанием ?p=1, ?p=2, ... (по умолчанию)
  saved_pages  HTML-страницы, сохранённые вручную в mcp_server/pages/<категория>*.html
               (если сайт недоступен или CATALOG_OFFLINE=true)
Демо-товаров нет: если нет ни живых, ни сохранённых данных, поиск честно возвращает ошибку.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# MCP-сервер запускается отдельным процессом: подхватываем .env из корня проекта сами
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("CATALOG_BASE_URL", "https://www.velosklad.ru")
OFFLINE = os.getenv("CATALOG_OFFLINE", "false").lower() == "true"
PAGES_DIR = Path(__file__).parent / "pages"
MAX_PAGES = int(os.getenv("CATALOG_MAX_PAGES", "10"))       # страниц на раздел (по ~21 товару)
PAGE_DELAY = float(os.getenv("CATALOG_PAGE_DELAY", "0.5"))  # пауза между страницами, секунд
CACHE_TTL = 600                                              # сек: повторные поиски без запросов к сайту

CATEGORY_PATHS = {
    "mountain": "/velosipedy/type/gornye/",
    "road": "/velosipedy/type/shosseynyie/",
    "city": "/velosipedy/type/dorozhnyie/",
    "hybrid": "/velosipedy/type/gornye-comfort/",
    "kids": "/velosipedy/type/detskie/",
    "teen": "/velosipedy/type/podrostkovyie/",
    "folding": "/velosipedy/type/skladnyie/",
    "full_suspension": "/velosipedy/type/dvuhpodvesy/",
}

# Вёрстка velosklad.ru: карточка = div.ah-products-list__col (разметка schema.org).
CARD_SELECTOR = ".ah-products-list__col"
NAME_SELECTOR = "a.ah-products-item__name"
PRICE_META = "meta[itemprop='price']"
STOCK_SELECTOR = ".ah-products-item__stock"
SPEC_ROW = ".ah-characteristics__row"
BIKE_PATH = "/velosipedy/bike/"

SPEC_KEYS = {  # название характеристики на сайте -> ключ у нас
    "диаметр колес": "wheel_size",
    "тип тормозов": "brakes",
    "тип рамы": "frame",
    "тип вилки": "fork",
    "скорости": "speeds",
    "вес": "weight",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}

_memory: dict[str, tuple[float, str, list[dict]]] = {}  # category -> (время, источник, товары)


def _parse_wheel(value: str) -> str | None:
    m = re.search(r"(\d{2}(?:[.,]\d)?)", value or "")
    return m.group(1).replace(",", ".") if m else None


def _specs_from_card(card) -> dict:
    specs: dict = {}
    for row in card.select(SPEC_ROW):
        label = row.select_one(".ah-characteristics__text")
        value = row.select_one(".ah-characteristics__value")
        if not label or not value:
            continue
        key = SPEC_KEYS.get(label.get_text(" ", strip=True).rstrip(":").strip().lower())
        text = value.get_text(" ", strip=True)
        if not key or not text:
            continue
        if key == "wheel_size":
            text = _parse_wheel(text)
            if not text:
                continue
        specs[key] = text.lower() if key in ("brakes", "frame", "fork") else text
    return specs


def parse_listing(html: str, category: str, source: str = "live") -> list[dict]:
    """Разобрать страницу раздела: товары с названием, ценой, ссылкой и характеристиками."""
    soup = BeautifulSoup(html, "lxml")
    products: list[dict] = []
    seen: set[str] = set()
    for card in soup.select(CARD_SELECTOR):
        link = card.select_one(NAME_SELECTOR)
        if link is None:
            continue
        href = link.get("href", "").split("?")[0].split("#")[0].strip()
        if BIKE_PATH not in href:
            continue
        url = urljoin(BASE_URL, href)
        name = link.get_text(" ", strip=True)
        if url in seen or len(name) < 4:
            continue
        seen.add(url)

        price_el = card.select_one(PRICE_META)
        try:
            price = int(float(price_el.get("content", ""))) if price_el else None
        except ValueError:
            price = None
        stock_el = card.select_one(STOCK_SELECTOR)
        stock = stock_el.get_text(" ", strip=True) if stock_el else ""
        specs = _specs_from_card(card)
        m_year = re.search(r"\((20\d{2})\)", name)
        if m_year:
            specs["year"] = int(m_year.group(1))
        products.append(
            {
                "name": name,
                "price": price,
                "currency": "RUB",
                "url": url,
                "category": category,
                "specs": specs,
                "stock": stock,
                "source": source,
            }
        )
    return products


def _fetch_live(category: str) -> list[dict]:
    """Листаем ?p=1, ?p=2, ... пока на странице появляются новые товары (но не больше MAX_PAGES)."""
    path = CATEGORY_PATHS.get(category)
    if not path:
        return []
    base = urljoin(BASE_URL, path)
    items: list[dict] = []
    seen: set[str] = set()
    with httpx.Client(headers=HEADERS, timeout=20.0, follow_redirects=True) as client:
        for page in range(1, MAX_PAGES + 1):
            resp = client.get(base, params={"p": page} if page > 1 else None)
            resp.raise_for_status()
            fresh = [p for p in parse_listing(resp.text, category) if p["url"] not in seen]
            if not fresh:
                break  # страницы закончились (или сайт отдал ту же страницу)
            seen.update(p["url"] for p in fresh)
            items.extend(fresh)
            time.sleep(PAGE_DELAY)
    return items


def _load_saved(category: str) -> list[dict]:
    """Страницы, сохранённые вручную: mcp_server/pages/<категория>.html, <категория>_2.html, ..."""
    items: list[dict] = []
    seen: set[str] = set()
    for f in sorted(PAGES_DIR.glob(f"{category}*.html")):
        html = f.read_text(encoding="utf-8", errors="ignore")
        for p in parse_listing(html, category, source="saved_pages"):
            if p["url"] not in seen:
                seen.add(p["url"])
                items.append(p)
    return items


def load_category(category: str) -> tuple[list[dict], str, str]:
    """Товары раздела: (товары, источник, пояснение). Источник: live / saved_pages / error."""
    cached = _memory.get(category)
    if cached and time.time() - cached[0] < CACHE_TTL:
        return cached[2], cached[1], ""
    note = ""
    if not OFFLINE:
        try:
            items = _fetch_live(category)
            if items:
                _memory[category] = (time.time(), "live", items)
                return items, "live", ""
            note = "Сайт ответил, но карточки не найдены (возможно, сменилась вёрстка)."
        except Exception as exc:  # сеть, 4xx/5xx, таймаут
            note = f"Сайт недоступен ({type(exc).__name__})."
    items = _load_saved(category)
    if items:
        _memory[category] = (time.time(), "saved_pages", items)
        return items, "saved_pages", note
    return [], "error", (note + " " if note else "") + (
        f"Нет и сохранённых страниц: положите HTML раздела в {PAGES_DIR.name}/{category}.html."
    )


def _matches(p: dict, category: str, max_price: int | None, wheel_size: str | None) -> bool:
    if category and p.get("category") != category:
        return False
    if max_price is not None:
        price = p.get("price")
        if price is None or price > max_price:
            return False
    if wheel_size:
        norm_target = str(wheel_size).replace(",", ".").strip()
        item_wheel = str(p.get("specs", {}).get("wheel_size", "")).replace(",", ".").strip()
        if item_wheel != norm_target:
            return False
    return True


def search(
    category: str,
    max_price: int | None = None,
    wheel_size: str | None = None,
    limit: int = 10,
) -> dict:
    """Поиск товаров. Возвращает данные и признак источника (live / saved_pages / error)."""
    items, source, note = load_category(category)
    found = [p for p in items if _matches(p, category, max_price, wheel_size)]
    found.sort(key=lambda p: (p.get("price") is None, p.get("price") or 0))
    return {
        "source": source,
        "note": note,
        "total_in_category": len(items),
        "count": len(found[:limit]),
        "items": found[:limit],
    }


def get_product(url: str) -> dict:
    """Карточка товара по ссылке из уже загруженных разделов (отдельную страницу товара не парсим)."""
    url = url.split("?")[0].split("#")[0]
    for _, _, items in _memory.values():
        for p in items:
            if p.get("url") == url:
                return p
    for category in CATEGORY_PATHS:
        for p in _load_saved(category):
            if p.get("url") == url:
                return p
    return {"error": f"Товар {url} не найден среди загруженных разделов: сначала выполните search_catalog"}


def calculate_total(items: list[dict], quantity: int = 1) -> dict:
    """Расчёт стоимости: цена за единицу × количество (скидки не применяются)."""
    if quantity < 1:
        return {"error": "quantity должен быть >= 1"}
    lines = []
    total = 0
    for it in items:
        price = it.get("price")
        if price is None:
            continue
        line = price * quantity
        total += line
        lines.append(
            {
                "name": it.get("name"),
                "unit_price": price,
                "quantity": quantity,
                "line_total": line,
            }
        )
    return {
        "lines": lines,
        "total": total,
        "currency": "RUB",
    }
