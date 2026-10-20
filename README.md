# Мультиагентная система подбора велосипедов (LangGraph)

Заявка клиента -> supervisor оркестрирует агентов -> коммерческое предложение.

## Архитектура

```
START -> start_turn -> supervisor <-> analyst | searcher | checker | advisor | writer
                           \-> finalize -> END
```

| Агент | Модель | Что делает |
|---|---|---|
| supervisor | Gemini | выбирает следующего агента из допустимых (допустимые шаги считает код) |
| analyst | Qwen (Ollama) | текст заявки -> структурированные требования (Pydantic) |
| searcher | Gemini + MCP | вызывает инструменты своего MCP-сервера каталога |
| checker | Gemini | проверка обязательных требований и пожеланий |
| advisor | Gemini + RAG | консультации и условия поставки из базы знаний |
| writer | Gemini + MCP | КП: числа из `calculate_total`, текст пишет модель |

- **MCP**: `mcp_server/` (свой сервер: `search_catalog`, `get_product`, `calculate_total`).
- **RAG**: `kb/` -> Chroma, инструмент `kb_search`.
- **Память**: краткосрочная (чекпоинтер SQLite по `thread_id`) и долговременная (профиль клиента, `app/memory.py`).

## Запуск

```
pip install -r requirements.txt
copy .env.example .env            # вставьте GOOGLE_API_KEY
ollama pull qwen2.5:7b
ollama pull bge-m3
python -m app.rag.ingest          # один раз: индексация базы знаний
python -m app.main                # чат
python -m app.main --graph        # схема графа (Mermaid) для пояснительной записки
python -m unittest discover -s tests -v   # тесты без LLM и сети
```

Все команды выполняйте из корня проекта (там, где лежит папка `app`).

## Сценарии для проверки и видео

1. «Нужен горный велосипед, колёса 29, до 20 000 рублей, недорогой»: полное КП.
2. «Горный до 5000 рублей»: нет подходящих, система предлагает изменить бюджет.
3. «Хочу велосипед»: уточняющий вопрос; ответ «горный» в том же диалоге продолжает заявку.
4. «Какая у вас гарантия?»: ответ из базы знаний (RAG).
5. `/new` и затем «как в прошлый раз»: работает долговременная память.
6. «Нужно 5 горных, до 20 000»: оптовая скидка из `calculate_total`.

## Что подстроить

- `CATEGORY_PATHS` и `SELECTORS` в `mcp_server/catalog.py` под реальную вёрстку сайта.
- Товары в `catalog_cache.json` и условия в `kb/` учебные.
