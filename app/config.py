"""Единая точка чтения настроек из .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else (ROOT / p).resolve()


LLM_LIGHT_PROVIDER = os.getenv("LLM_LIGHT_PROVIDER", "ollama")
LLM_LIGHT_MODEL = os.getenv("LLM_LIGHT_MODEL", "qwen2.5:7b")
LLM_HEAVY_PROVIDER = os.getenv("LLM_HEAVY_PROVIDER", "gemini")
LLM_HEAVY_MODEL = os.getenv("LLM_HEAVY_MODEL", "gemini-2.5-flash")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

CATALOG_BASE_URL = os.getenv("CATALOG_BASE_URL", "https://velosklad.ru")
CATALOG_OFFLINE = os.getenv("CATALOG_OFFLINE", "false").lower() == "true"

CHROMA_DIR = _path("CHROMA_DIR", "./data/chroma")
CHECKPOINT_DB = _path("CHECKPOINT_DB", "./data/checkpoints.sqlite")
STORE_DB = _path("STORE_DB", "./data/store.sqlite")
KB_DIR = ROOT / "kb"
