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
from dotenv import load_dotenv

# MCP-сервер запускается отдельным процессом: подхватываем .env из корня проекта сами
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("CATALOG_BASE_URL", "https://www.velosklad.ru")
OFFLINE = os.getenv("CATALOG_OFFLINE", "false").lower() == "true"
CACHE_PATH = Path(__file__).parent / "data" / "catalog_cache.json"

# Реальные пути категорий на сайте velosklad.ru
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

# Признак ссылки на карточку товара. Карточки ищутся ПО ЭТИМ ссылкам, а не по классам-оберткам:
# так в результат не попадают фильтры, меню и баннеры. Если после изменений поиск всегда
# уходит в кэш, сначала проверьте, что реальные ссылки товаров содержат именно эту часть пути.
MODEL_PATH = "/velosipedy/model/"

# Селекторы проверяются ПО ОЧЕРЕДИ, сверху вниз (приоритет). Раньше они были одним списком
# через запятую, но select_one возвращает первый элемент в порядке документа, и приоритет
# не работал: брался внешний блок .price со старой и новой ценой вместе.
TITLE_SELECTORS = [
    ".product-title", ".item-title", ".name",
    "[class*='title']", "[class*='name']", "h3", "h2",
]
PRICE_SELECTORS = [
    ".price-new", ".current-price", ".product-price", ".item-price",
    ".price", "[class*='price']",
]
STOP_TITLES = {"подробнее", "купить", "сравнить"}

# Стандартный User-Agent браузера, чтобы не срабатывала антибот-защита
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _load_cache() -> list[dict]:
    if not CACHE_PATH.exists():
        return []
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _first(node, selectors: list[str]):
    """Первый найденный элемент по списку селекторов с учётом приоритета."""
    for sel in selectors:
        el = node.select_one(sel)
        if el is not None:
            return el
    return None


# Сумма: группы по 3 цифры через пробел/nbsp («25 990») или просто 4-7 цифр («25990»).
# (?<!\d) не даёт начать число из середины другого числа (артикул + цена не склеиваются).
_AMOUNT = r"(?<!\d)(\d{1,3}(?:[\s\xa0]\d{3})+|\d{4,7})"
_RUB = re.compile(_AMOUNT + r"\s*(?:₽|руб|р\.)", re.IGNORECASE)
_BARE = re.compile(_AMOUNT + r"(?!\d)")


def _parse_price(text: str) -> int | None:
    """Актуальная цена в рублях. Если в тексте рядом старая и новая цена, берём меньшую."""
    if not text:
        return None
    for pattern in (_RUB, _BARE):  # сначала суммы со знаком рубля, затем «голые» числа
        values = [int(re.sub(r"\D", "", raw)) for raw in pattern.findall(text)]
        values = [v for v in values if 1000 <= v <= 9_999_999]
        if values:
            return min(values)
    return None


def _extract_specs(title: str) -> dict:
    """Достаём характеристики из названия (диаметр колёс и модельный год)."""
    specs: dict = {}
    if not title:
        return specs

    # 1. Ищем диаметр с кавычками или словом "дюйм" (например: 29", 27.5 дюймов)
    m_explicit = re.search(r'(\d{2}(?:[.,]\d)?)\s*(?:"|”|дюйм|\'\')', title, re.IGNORECASE)
    if m_explicit:
        specs["wheel_size"] = m_explicit.group(1).replace(",", ".")
    else:
        # 2. На velosklad.ru диаметр пишется прямо в названии: "Forward Sporting 29 (2026)"
        m_size = re.search(
            r"\b(12|14|16|18|20|24|26|27[.,]5|28|29|32)\b(?!\s*(?:ск|скорост|г\.|год))",
            title,
        )
        if m_size:
            specs["wheel_size"] = m_size.group(1).replace(",", ".")

    # Дополнительно извлекаем год модели в скобках, если есть: "(2025)" или "(2026)"
    m_year = re.search(r"\((20\d{2})\)", title)
    if m_year:
        specs["year"] = int(m_year.group(1))

    return specs


def _model_links(node):
    return node.select(f"a[href*='{MODEL_PATH}']")


def _find_card(link):
    """Поднимаемся от ссылки на модель к ближайшему контейнеру, где есть цена.

    Если в контейнере уже несколько разных товаров, значит поднялись слишком высоко.
    """
    node = link.parent
    for _ in range(6):
        if node is None or node.name in ("body", "html", "[document]"):
            return None
        urls = {a.get("href", "").split("#")[0] for a in _model_links(node)}
        if len(urls) > 1:
            return None
        if _first(node, PRICE_SELECTORS) is not None:
            return node
        node = node.parent
    return None


def parse_listing(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    products: list[dict] = []
    seen_urls: set[str] = set()

    for link in _model_links(soup):
        href = link.get("href", "").strip()
        url = urljoin(BASE_URL, href.split("#")[0])
        if url in seen_urls:
            continue

        card = _find_card(link)
        if card is None:
            continue

        # Название: самый длинный текст среди ссылок на эту модель (ссылка-картинка пустая),
        # иначе заголовок карточки.
        title = max((a.get_text(" ", strip=True) for a in _model_links(card)), key=len, default="")
        if len(title) < 4 or title.lower() in STOP_TITLES:
            el = _first(card, TITLE_SELECTORS)
            title = el.get_text(" ", strip=True) if el else ""
        if len(title) < 4 or title.lower() in STOP_TITLES:
            continue

        price_el = _first(card, PRICE_SELECTORS)
        price = _parse_price(price_el.get_text(" ", strip=True)) if price_el else None

        seen_urls.add(url)
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
    target_url = urljoin(BASE_URL, path)
    with httpx.Client(headers=HEADERS, timeout=15.0, follow_redirects=True) as client:
        resp = client.get(target_url)
        resp.raise_for_status()
    return parse_listing(resp.text, category)


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
    """Поиск товаров. Возвращает данные и признак источника (live / cache)."""
    source = "cache"
    items: list[dict] = []
    note = ""
    if not OFFLINE:
        try:
            items = _fetch_live(category)
            if items:
                source = "live"
            else:
                note = "Сайт ответил, но карточки не найдены (возможно, сменилась вёрстка). Использован кэш."
        except Exception as exc:  # сеть, 4xx/5xx, таймаут
            note = f"Сайт недоступен ({type(exc).__name__}). Использован кэш."
    if not items:
        items = _load_cache()
        source = "cache"

    found = [p for p in items if _matches(p, category, max_price, wheel_size)]
    found.sort(key=lambda p: (p.get("price") is None, p.get("price") or 0))
    return {
        "source": source,
        "note": note,
        "count": len(found[:limit]),
        "items": found[:limit],
    }


def get_product(url: str) -> dict:
    """Карточка товара по URL (сначала ищет в кэше, затем пробует спарсить вживую)."""
    for p in _load_cache():
        if p.get("url") == url:
            return p

    if not OFFLINE and url.startswith("http"):
        try:
            with httpx.Client(headers=HEADERS, timeout=15.0, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            h1 = soup.find("h1")
            title = h1.get_text(" ", strip=True) if h1 else "Неизвестная модель"
            price_el = _first(soup, PRICE_SELECTORS)
            price = _parse_price(price_el.get_text(" ", strip=True)) if price_el else None
            return {
                "name": title,
                "price": price,
                "currency": "RUB",
                "url": url,
                "specs": _extract_specs(title),
                "source": "live",
            }
        except Exception as exc:
            return {"error": f"Не удалось загрузить карточку {url}: {exc}"}

    return {"error": f"Товар с URL {url} не найден"}


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
