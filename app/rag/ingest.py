"""Индексация базы знаний (kb/*.md) в Chroma.

Запуск:  python -m app.rag.ingest
Нужна запущенная Ollama с моделью эмбеддингов (ollama pull bge-m3).
"""
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .. import config
from ..llm import get_embeddings

COLLECTION = "bike_kb"


def _split_markdown(text: str, source: str) -> list[Document]:
    """Режем документ по заголовкам '## ' (каждый раздел = один фрагмент)."""
    title = text.splitlines()[0].lstrip("# ").strip() if text else source
    docs: list[Document] = []
    chunk: list[str] = []
    section = title

    def flush():
        body = "\n".join(chunk).strip()
        if len(body) > 30:
            docs.append(
                Document(
                    page_content=f"{title} / {section}\n{body}",
                    metadata={"source": source, "section": section},
                )
            )

    for line in text.splitlines()[1:]:
        if line.startswith("## "):
            flush()
            chunk = []
            section = line[3:].strip()
        else:
            chunk.append(line)
    flush()
    return docs


def build_index(reset: bool = True) -> int:
    docs: list[Document] = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        docs.extend(_split_markdown(path.read_text(encoding="utf-8"), path.name))
    if not docs:
        raise SystemExit(f"В {config.KB_DIR} нет .md файлов")

    if reset and config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
    )
    return len(docs)


if __name__ == "__main__":
    n = build_index()
    print(f"Проиндексировано фрагментов: {n}")
