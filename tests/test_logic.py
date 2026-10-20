"""Тесты чистой логики и памяти (без LLM и сети).

Запуск из корня проекта:  python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import logic, memory  # noqa: E402

CACHE = json.loads(
    (Path(__file__).resolve().parent.parent / "mcp_server" / "data" / "catalog_cache.json").read_text(encoding="utf-8")
)
MOUNTAIN = [p for p in CACHE if p["category"] == "mountain"]


class TestFilters(unittest.TestCase):
    def test_hard_filter_budget_and_wheels(self):
        req = {"category": "mountain", "max_price": 20000, "wheel_size": "29"}
        ok, rejected = logic.hard_filter(MOUNTAIN, req)
        self.assertEqual({p["price"] for p in ok}, {17990, 19490})
        reasons = " ".join(r for item in rejected for r in item["reasons"])
        self.assertIn("дороже бюджета", reasons)
        self.assertIn("колёса 26″ вместо 29″", reasons)

    def test_wheel_normalization(self):
        self.assertEqual(logic.normalize_wheel('29"'), "29")
        self.assertEqual(logic.normalize_wheel("27,5"), "27.5")
        self.assertEqual(logic.normalize_wheel("29.0"), "29")

    def test_other_category_rejected(self):
        ok, rejected = logic.hard_filter(CACHE, {"category": "kids"})
        self.assertTrue(all(p["category"] == "kids" for p in ok))
        self.assertTrue(rejected)

    def test_rank_drops_unsuitable_and_sorts(self):
        a, b, c = MOUNTAIN[0], MOUNTAIN[1], MOUNTAIN[2]
        checks = [
            {"url": a["url"], "suitable": True, "score": 6, "met": [], "unmet": []},
            {"url": b["url"], "suitable": True, "score": 9, "met": [], "unmet": []},
            {"url": c["url"], "suitable": False, "score": 1, "met": [], "unmet": ["дисковые тормоза"]},
        ]
        ranked, dropped = logic.rank_candidates([a, b, c], checks)
        self.assertEqual([r["url"] for r in ranked], [b["url"], a["url"]])
        self.assertEqual(len(dropped), 1)
        self.assertIn("дисковые тормоза", dropped[0]["reasons"][0])

    def test_missing_check_keeps_product(self):
        ranked, dropped = logic.rank_candidates([MOUNTAIN[0]], [])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(dropped, [])


class TestExtract(unittest.TestCase):
    def test_extract_products_dedup_and_meta(self):
        payload = {"source": "cache", "note": "n", "items": MOUNTAIN[:2]}
        msgs = [
            SimpleNamespace(name="search_catalog", content=json.dumps(payload, ensure_ascii=False)),
            SimpleNamespace(name="search_catalog", content=[{"type": "text", "text": json.dumps(payload)}]),
            SimpleNamespace(name="other_tool", content="{}"),
        ]
        products, meta = logic.extract_products(msgs)
        self.assertEqual(len(products), 2)
        self.assertEqual(meta["source"], "cache")

    def test_extract_handles_result_wrapper_and_garbage(self):
        wrapped = {"result": {"source": "live", "items": MOUNTAIN[:1]}}
        msgs = [
            SimpleNamespace(name="search_catalog", content=json.dumps(wrapped)),
            SimpleNamespace(name="search_catalog", content="ошибка инструмента"),
        ]
        products, meta = logic.extract_products(msgs)
        self.assertEqual(len(products), 1)
        self.assertEqual(meta["source"], "live")

    def test_parse_tool_json_with_noise(self):
        self.assertEqual(logic.parse_tool_json('текст {"a": 1} хвост'), {"a": 1})
        self.assertIsNone(logic.parse_tool_json("нет json"))


class TestRouting(unittest.TestCase):
    def test_order_flow(self):
        s = {}
        self.assertEqual(logic.allowed_routes(s), ["analyst"])
        s.update(analyzed=True, intent="order")
        self.assertEqual(logic.allowed_routes(s), ["searcher"])
        s.update(searched=True, candidates=[1])
        self.assertEqual(logic.allowed_routes(s), ["checker"])
        s.update(checked=True, ranked=[1])
        self.assertEqual(logic.allowed_routes(s), ["advisor", "writer"])
        s.update(advised=True)
        self.assertEqual(logic.allowed_routes(s), ["writer"])
        s.update(written=True)
        self.assertEqual(logic.allowed_routes(s), ["END"])

    def test_no_candidates_goes_to_writer(self):
        s = {"analyzed": True, "intent": "order", "searched": True, "candidates": []}
        self.assertEqual(logic.allowed_routes(s), ["writer"])

    def test_consult_and_other(self):
        self.assertEqual(logic.allowed_routes({"analyzed": True, "intent": "consult"}), ["advisor"])
        self.assertEqual(logic.allowed_routes({"analyzed": True, "intent": "consult", "advised": True}), ["END"])
        self.assertEqual(logic.allowed_routes({"analyzed": True, "intent": "other"}), ["END"])

    def test_clarification_ends_turn(self):
        self.assertEqual(logic.allowed_routes({"analyzed": True, "intent": "order", "answer": "Какой тип?"}), ["END"])


class TestTexts(unittest.TestCase):
    def test_no_results_suggests_budget(self):
        req = {"category": "mountain", "max_price": 10000, "quantity": 1, "must_have": []}
        _, rejected = logic.hard_filter(MOUNTAIN, req)
        text = logic.no_results_text(req, rejected, {"source": "cache"})
        self.assertIn("не нашлось", text)
        self.assertIn("увеличить бюджет до 13 990 ₽", text)
        self.assertIn("кэша", text)

    def test_offer_uses_data_not_model_text(self):
        req = {"category": "mountain", "max_price": 20000, "quantity": 5, "wishes": ["недорогой"]}
        p = {**MOUNTAIN[0], "check": {"met": ["недорогой"], "unmet": []}}
        rows = [{"product": p, "calc": {"total": 87252, "discount_percent": 3}}]
        text = logic.render_offer(req, rows, {"intro": "Здравствуйте!"}, date(2026, 10, 20), {"source": "cache"})
        self.assertIn("20.10.2026", text)
        self.assertIn("17 990 ₽ за единицу", text)
        self.assertIn("скидка 3%", text)
        self.assertIn("87 252 ₽", text)
        self.assertIn(MOUNTAIN[0]["url"], text)
        self.assertIn("кэша", text)


class TestMemory(unittest.TestCase):
    def test_profile_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "store.sqlite"
            self.assertEqual(memory.load_profile("u1", db), {})
            memory.update_profile("u1", {"category": "mountain", "max_price": 20000, "wishes": ["недорогой"]}, db)
            prof = memory.update_profile("u1", {"category": "city", "wishes": ["недорогой", "лёгкий"]}, db)
            self.assertEqual(prof["offers"], 2)
            self.assertEqual(prof["last_category"], "city")
            self.assertEqual(prof["last_max_price"], 20000)
            self.assertEqual(prof["wishes"], ["недорогой", "лёгкий"])
            self.assertEqual(memory.load_profile("u2", db), {})
            self.assertIn("city", memory.profile_to_text(prof))


if __name__ == "__main__":
    unittest.main()
