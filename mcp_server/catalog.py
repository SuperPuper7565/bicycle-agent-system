"""Логика каталога: загрузка страницы, детерминированный парсинг, локальный кэш.

Парсинг выполняется обычным кодом (BeautifulSoup), без участия LLM.
Если сайт недоступен, изменилась вёрстка или включён CATALOG_OFFLINE,
используется локальный снимок data/catalog_cache.json.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

BASE_URL = os.getenv("CATALOG_BASE_URL", "https://velosklad.ru")
OFFLINE = os.getenv("CATALOG_OFFLINE", "false").lower() == "true"
CACHE_PATH = Path(__file__).parent / "data" / "catalog_cache.json"

# Категории -> путь на сайте. ПРОВЕРЬТЕ и поправьте под реальный сайт.
CATEGORY_PATHS = {
    "mountain": "/catalog/gornye-velosipedy/",
    "road": "/catalog/shosseynye-velosipedy/",
    "city": "/catalog/gorodskie-velosipedy/",
    "hybrid": "/catalog/gibridnye-velosipedy/",
    "kids": "/catalog/detskie-velosipedy/",
}

# CSS-селекторы карточки товара. ПОДСТРОЙТЕ под реальную вёрстку:
# откройте страницу категории, F12 -> Elements, найдите контейнер карточки.
SELECTORS = {
    "card": ".product-card, .catalog-item, .product-item",
    "title": ".product-title, .item-title, h3, h2",
    "price": ".price, .product-price, .item-price",
    "link": "a[href]",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (student-lab; +langgraph-agents)"}


def _load_cache() -> list[dict]:
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _parse_price(text: str) -> int | None:
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else None


def _extract_specs(title: str) -> dict:
    """Достаём простые характеристики из названия (диаметр колёс и т.п.)."""
    specs: dict = {}
    m = re.search(r'(\d{2}(?:[.,]\d)?)\s*(?:"|”|дюйм|\'\')', title)
    if m:
        specs["wheel_size"] = m.group(1).replace(",", ".")
    return specs


def parse_listing(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    products: list[dict] = []
    for card in soup.select(SELECTORS["card"]):
        title_el = card.select_one(SELECTORS["title"])
        price_el = card.select_one(SELECTORS["price"])
        link_el = card.select_one(SELECTORS["link"])
        if not title_el:
            continue
        title = title_el.get_text(" ", strip=True)
        price = _parse_price(price_el.get_text(" ", strip=True)) if price_el else None
        url = urljoin(BASE_URL, link_el["href"]) if link_el else None
        products.append(
            {
                "name": title,
                "price": price,
                "currency": "RUB",
                "url": url,
                "category": category,
                "specs": _extract_specs(title),
                "source": "live",
            }
        )
    return products


def _fetch_live(category: str) -> list[dict]:
    path = CATEGORY_PATHS.get(category)
    if not path:
        return []
    with httpx.Client(headers=HEADERS, timeout=15, follow_redirects=True) as client:
        resp = client.get(urljoin(BASE_URL, path))
        resp.raise_for_status()
    return parse_listing(resp.text, category)


def _matches(p: dict, category: str, max_price: int | None, wheel_size: str | None) -> bool:
    if category and p.get("category") != category:
        return False
    if max_price is not None and p.get("price") is not None and p["price"] > max_price:
        return False
    if wheel_size and p.get("specs", {}).get("wheel_size") != wheel_size:
        return False
    return True


def search(category: str, max_price: int | None = None, wheel_size: str | None = None,
           limit: int = 10) -> dict:
    """Поиск товаров. Возвращает данные и признак источника (live / cache)."""
    source = "cache"
    items: list[dict] = []
    note = ""
    if not OFFLINE:
        try:
            items = _fetch_live(category)
            source = "live"
            if not items:
                note = "Сайт ответил, но карточки не найдены (возможно, сменилась вёрстка). Использован кэш."
        except Exception as exc:  # сеть, 4xx/5xx, таймаут
            note = f"Сайт недоступен ({type(exc).__name__}). Использован кэш."
    if not items:
        items = _load_cache()
        source = "cache"
    found = [p for p in items if _matches(p, category, max_price, wheel_size)]
    found.sort(key=lambda p: (p.get("price") is None, p.get("price") or 0))
    return {"source": source, "note": note, "count": len(found[:limit]), "items": found[:limit]}


def get_product(url: str) -> dict:
    """Карточка товара по URL (из кэша; для живого режима можно расширить)."""
    for p in _load_cache():
        if p.get("url") == url:
            return p
    return {"error": f"Товар с URL {url} не найден"}


def calculate_total(items: list[dict], quantity: int = 1) -> dict:
    """Расчёт стоимости: цена за единицу, сумма, скидка за опт."""
    if quantity < 1:
        return {"error": "quantity должен быть >= 1"}
    discount = 0.0
    if quantity >= 10:
        discount = 0.07
    elif quantity >= 5:
        discount = 0.03
    lines = []
    total = 0
    for it in items:
        price = it.get("price")
        if price is None:
            continue
        line = round(price * quantity * (1 - discount))
        total += line
        lines.append({"name": it.get("name"), "unit_price": price, "quantity": quantity, "line_total": line})
    return {"lines": lines, "discount_percent": int(discount * 100), "total": total, "currency": "RUB"}
