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
    (Path(__file__).resolve().parent / "fixtures" / "demo_products.json").read_text(encoding="utf-8")
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
        text = logic.no_results_text(req, rejected, {"source": "saved_pages"})
        self.assertIn("не нашлось", text)
        self.assertIn("увеличить бюджет до 13 990 ₽", text)
        self.assertIn("сохранённых страниц", text)

    def test_offer_uses_data_not_model_text(self):
        req = {"category": "mountain", "max_price": 20000, "quantity": 5, "wishes": ["недорогой"]}
        p = {**MOUNTAIN[0], "check": {"met": ["недорогой"], "unmet": []}}
        rows = [{"product": p, "calc": {"total": 89950}}]
        text = logic.render_offer(req, rows, {"intro": "Здравствуйте!", "advice": "- Размер рамы подбирают по росту"}, date(2026, 10, 20), {"source": "saved_pages"})
        self.assertIn("20.10.2026", text)
        self.assertIn("17 990 ₽ за единицу", text)
        self.assertNotIn("скидка", text)
        self.assertIn("89 950 ₽", text)
        self.assertIn("На что обратить внимание", text)
        self.assertIn("Размер рамы", text)
        self.assertIn(MOUNTAIN[0]["url"], text)
        self.assertIn("сохранённых страниц", text)


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


class TestTotals(unittest.TestCase):
    def test_total_is_price_times_quantity_without_discount(self):
        from mcp_server import catalog

        r = catalog.calculate_total([{"name": "a", "price": 10000}, {"name": "b", "price": 20000}], 10)
        self.assertEqual(r["total"], 300000)
        self.assertNotIn("discount_percent", r)
        self.assertIn("error", catalog.calculate_total([], 0))


class TestRequirementsPolicy(unittest.TestCase):
    def test_numbers_in(self):
        nums = logic.numbers_in("до двадцати пяти тысяч, колёса 27,5, пять штук, 150 000, 20к, 20 тысяч")
        for value in (25000, 27.5, 5, 150000, 20000):
            self.assertIn(float(value), nums)

    def test_ground_drops_invented_values(self):
        new = {"category": "mountain", "max_price": 15000, "wheel_size": "29",
               "wishes": ["крутой"], "for_child": True, "quantity": None}
        out = logic.ground_requirements(new, "хочу велик сынуле купить")
        self.assertIsNone(out["category"])
        self.assertIsNone(out["max_price"])
        self.assertIsNone(out["wheel_size"])
        self.assertEqual(out["wishes"], [])
        self.assertTrue(out["for_child"])

    def test_ground_keeps_stated_values(self):
        text = "Нужен горный велосипед, колёса 29, до 20 000 рублей, недорогой"
        out = logic.ground_requirements(
            {"category": "mountain", "max_price": 20000, "wheel_size": "29", "wishes": ["недорогой"]}, text)
        self.assertEqual((out["category"], out["max_price"], out["wheel_size"]), ("mountain", 20000, "29"))
        self.assertEqual(out["wishes"], ["недорогой"])

    def test_ground_total_budget_for_several_bikes(self):
        out = logic.ground_requirements({"max_price": 20000, "quantity": 5}, "5 велосипедов, всего 100 000 рублей")
        self.assertEqual(out["max_price"], 20000)

    def test_new_order_does_not_inherit_previous(self):
        prev = {"category": "mountain", "max_price": 15000, "wheel_size": "29", "wishes": ["крутой"]}
        merged, _, carried = logic.merge_requirements(
            prev, {"for_child": True}, awaiting=False, text="хочу велик сынуле",
            profile={"last_category": "mountain"})
        self.assertFalse(carried)
        self.assertEqual(merged["category"], "kids")
        self.assertIsNone(merged.get("max_price"))
        self.assertIsNone(merged.get("wheel_size"))

    def test_answer_to_question_continues_previous(self):
        prev = {"category": "kids", "for_child": True}
        merged, _, carried = logic.merge_requirements(prev, {"child_age": 8}, awaiting=True, text="8 лет", profile={})
        self.assertTrue(carried)
        self.assertEqual((merged["category"], merged["child_age"]), ("kids", 8))

    def test_refinement_without_category_continues(self):
        prev = {"category": "mountain", "wheel_size": "29", "max_price": 20000}
        merged, _, carried = logic.merge_requirements(
            prev, {"max_price": 15000}, awaiting=False, text="а до 15 тысяч", profile={})
        self.assertTrue(carried)
        self.assertEqual((merged["wheel_size"], merged["max_price"]), ("29", 15000))

    def test_other_category_while_awaiting_starts_new_order(self):
        prev = {"category": "kids", "for_child": True}
        merged, _, carried = logic.merge_requirements(
            prev, {"category": "road", "max_price": 30000}, awaiting=True,
            text="нет, шоссейный до 30 тысяч", profile={})
        self.assertFalse(carried)
        self.assertFalse(merged["for_child"])
        self.assertEqual(merged["category"], "road")

    def test_profile_only_on_explicit_reference(self):
        profile = {"last_category": "mountain", "last_max_price": 20000, "last_wheel_size": "29"}
        m1, _, _ = logic.merge_requirements({}, {}, awaiting=False, text="как в прошлый раз", profile=profile)
        self.assertEqual((m1["category"], m1["max_price"], m1["wheel_size"]), ("mountain", 20000, "29"))
        m2, _, _ = logic.merge_requirements({}, {}, awaiting=False, text="хочу велосипед", profile=profile)
        self.assertIsNone(m2.get("category"))

    def test_next_question_only_when_needed(self):
        self.assertEqual(logic.next_question({}, False), logic.ASK_CATEGORY)
        self.assertEqual(logic.next_question({"category": "kids"}, False), logic.ASK_CHILD)
        self.assertIsNone(logic.next_question({"category": "kids", "child_age": 8}, False))
        self.assertIsNone(logic.next_question({"category": "kids", "wheel_size": "20"}, False))
        self.assertIsNone(logic.next_question({"category": "kids"}, True))
        self.assertIsNone(logic.next_question({"category": "mountain"}, False))


if __name__ == "__main__":
    unittest.main()


class FollowUpTests(unittest.TestCase):
    def test_search_command(self):
        from app.logic import wants_search_now
        for t in ("найди уже на сайте", "а велики-то ты будешь мне искать?", "подбери"):
            self.assertTrue(wants_search_now(t), t)
        self.assertFalse(wants_search_now("какой размер рамы"))

    def test_implausible_age(self):
        from app.logic import ground_requirements, merge_requirements, next_question, ASK_CHILD_AGAIN
        new = ground_requirements({"for_child": True, "child_age": 67}, "моему ребенку 67 лет")
        self.assertIsNone(new["child_age"])
        merged, _, _ = merge_requirements({"category": "kids", "for_child": True}, new,
                                          awaiting=True, text="", profile=None)
        self.assertEqual(next_question(merged, True), ASK_CHILD_AGAIN)

    def test_teen_and_category_reply(self):
        from app.logic import ground_requirements, merge_requirements, next_question
        prev = {"category": "kids", "for_child": True}
        new = ground_requirements({"for_child": True, "child_age": 15, "child_height_cm": 183},
                                  "моему ребенку 15 лет рост 183 см")
        merged, _, carried = merge_requirements(prev, new, awaiting=True, text="", profile=None)
        self.assertTrue(carried)
        self.assertEqual(merged["category"], "teen")
        self.assertIsNone(next_question(merged, True))
        new2 = ground_requirements({"category": "mountain"}, "пусть будет горный велик")
        m2, _, c2 = merge_requirements(merged, new2, awaiting=True, text="", profile=None)
        self.assertTrue(c2)
        self.assertEqual(m2["category"], "mountain")
        self.assertEqual(m2["child_age"], 15)


class ChildSizeParsingTests(unittest.TestCase):
    def test_garbage_answer_is_asked_again_even_if_model_returned_nothing(self):
        from app.logic import ground_requirements, merge_requirements, next_question, ASK_CHILD_AGAIN
        t = "моему ребенку 67 лет рост 1488 см"
        new = ground_requirements({"intent": "order", "for_child": True}, t)
        merged, _, _ = merge_requirements({"category": "kids", "for_child": True}, new,
                                          awaiting=True, text=t, profile=None)
        self.assertEqual(next_question(merged, True), ASK_CHILD_AGAIN)

    def test_age_parsed_without_model(self):
        from app.logic import ground_requirements
        new = ground_requirements({"for_child": True}, "сыну 12 лет, рост 150 см")
        self.assertEqual((new["child_age"], new["child_height_cm"]), (12.0, 150.0))


class ListingParserTests(unittest.TestCase):
    HTML = """
    <div class="ah-products-list__col" itemscope>
      <a href="/velosipedy/bike/28433/stels-navigator-620-v-26-k010/" class="ah-products-item__img"><img></a>
      <div class="ah-products-item__stock">В наличии</div>
      <a itemprop="url" href="/velosipedy/bike/28433/stels-navigator-620-v-26-k010/" class="ah-products-item__name">Велосипед Stels Navigator 620 V 26 K010 (2025)</a>
      <a href="/velosipedy/bike/28433/stels-navigator-620-v-26-k010/?addcomment=1#coment">11</a>
      <meta itemprop="price" content="19990">
      <div class="ah-products-item__old-price">24 990</div>
      <div class="ah-characteristics__row"><div class="ah-characteristics__text">Диаметр колес:</div><div class="ah-characteristics__value">26 дюймов</div></div>
      <div class="ah-characteristics__row"><div class="ah-characteristics__text">Тип тормозов:</div><div class="ah-characteristics__value">Ободные механические</div></div>
    </div>
    <div class="ah-products-list__col"><a href="/velosipedy/">Все велосипеды</a></div>
    """

    def test_real_layout(self):
        from mcp_server import catalog
        items = catalog.parse_listing(self.HTML, "mountain")
        self.assertEqual(len(items), 1)
        p = items[0]
        self.assertEqual(p["price"], 19990)  # не старая цена
        self.assertEqual(p["url"], "https://www.velosklad.ru/velosipedy/bike/28433/stels-navigator-620-v-26-k010/")
        self.assertEqual(p["specs"]["wheel_size"], "26")
        self.assertEqual(p["specs"]["brakes"], "ободные механические")
        self.assertEqual(p["stock"], "В наличии")

    def test_no_demo_fallback(self):
        from mcp_server import catalog
        catalog._memory.clear()
        old = (catalog.OFFLINE, catalog.PAGES_DIR)
        catalog.OFFLINE, catalog.PAGES_DIR = True, Path("/nonexistent")
        try:
            r = catalog.search("mountain")
        finally:
            catalog.OFFLINE, catalog.PAGES_DIR = old
        self.assertEqual((r["source"], r["count"]), ("error", 0))
