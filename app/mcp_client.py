"""Подключение агентов к СВОЕМУ MCP-серверу каталога (запускается подпроцессом по stdio)."""
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from . import config


async def load_mcp_tools() -> list:
    client = MultiServerMCPClient(
        {
            "catalog": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "env": {
                    "PYTHONPATH": str(config.ROOT),
                    "CATALOG_OFFLINE": "true" if config.CATALOG_OFFLINE else "false",
                    "CATALOG_BASE_URL": config.CATALOG_BASE_URL,
                },
            }
        }
    )
    return await client.get_tools()
