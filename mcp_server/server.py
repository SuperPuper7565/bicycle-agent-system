"""MCP-сервер каталога велосипедов.

Запуск (stdio, для LangGraph-клиента):   python -m mcp_server.server
Запуск по HTTP (для Docker / A2A-этапа): python -m mcp_server.server --http

Совместим с mcp 1.x (FastMCP) и mcp 2.x (MCPServer, переименованный FastMCP).
"""
import sys

try:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

    _V2 = False
except ModuleNotFoundError:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server

    _V2 = True

from . import catalog

mcp = _Server("bike-catalog")


@mcp.tool()
def search_catalog(category: str, max_price: int | None = None,
                   wheel_size: str | None = None, limit: int = 10) -> dict:
    """Найти велосипеды в каталоге.

    Args:
        category: тип велосипеда: mountain, road, city, hybrid, kids.
        max_price: максимальная цена за единицу в рублях.
        wheel_size: диаметр колёс в дюймах, например "29" или "27.5".
        limit: максимум результатов.

    Возвращает список товаров (название, цена, ссылка, характеристики)
    и поле source: live (сайт) или cache (локальный снимок).
    """
    return catalog.search(category, max_price, wheel_size, limit)


@mcp.tool()
def get_product(url: str) -> dict:
    """Получить карточку товара по ссылке."""
    return catalog.get_product(url)


@mcp.tool()
def calculate_total(items: list[dict], quantity: int = 1) -> dict:
    """Рассчитать итоговую стоимость выбранных товаров: цена × количество (скидки не применяются).

    Args:
        items: список товаров (поля name и price).
        quantity: количество единиц каждого товара.
    """
    return catalog.calculate_total(items, quantity)


def main() -> None:
    if "--http" in sys.argv:
        if _V2:
            mcp.run("streamable-http", host="0.0.0.0", port=8000)
        else:
            mcp.settings.host = "0.0.0.0"
            mcp.settings.port = 8000
            mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
