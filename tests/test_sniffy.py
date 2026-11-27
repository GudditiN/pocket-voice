"""SniffyNose lookup. No network: the site is replaced by small stand-in pages."""
import json
from types import SimpleNamespace as NS

import pytest

import live
import llm
import sniffy

S = "https://sniffynose.com"


def product(name, price, stock="InStock"):
    data = {"@type": "Product", "name": name, "image": ["https://cdn.example.com/a.png"],
            "offers": {"price": price, "availability": f"https://schema.org/{stock}"}}
    return f"<html><head><title>{name}</title><script type=\"application/ld+json\">{json.dumps(data)}</script></head></html>"


SITE = {
    "sitemap-products.xml": "".join(f"<loc>{S}/products/{p}</loc>" for p in
                                    ["chicken-dog-food", "puppy-dry-food", "cat-litter-box", "dog-rope-toy"]),
    "sitemap-static.xml": f"<loc>{S}/grooming</loc><loc>{S}/why-sniffynose</loc><loc>{S}/api/secret</loc>",
    "sitemap-categories.xml": f"<loc>{S}/products/categories/dog-food</loc>",
    "sitemap-brands.xml": f"<loc>{S}/products/brands/acana</loc>",
    "/products/chicken-dog-food": product("Chicken Dog Food", 1738),
    "/products/puppy-dry-food": product("Puppy Dry Food", 747, "OutOfStock"),
    "/products/dog-rope-toy": product("Dog Rope Toy", 199),
    "/grooming": "<title>Grooming</title><meta name='description' content='Book grooming'><body>Baths and trims</body>",
    "/why-sniffynose": "<title>Why</title><body>Pet care made simple</body>",
    f"{S}/": "<title>SniffyNose</title><body>Home</body>",
}


@pytest.fixture
def site(monkeypatch):
    seen = []

    def fake(url, *a, **k):
        seen.append(url)
        for part, body in sorted(SITE.items(), key=lambda kv: -len(kv[0])):
            if url.endswith(part):
                return 200, "text/html", body
        return 404, "text/html", ""

    monkeypatch.setattr(live, "fetch", fake)
    return seen


@pytest.mark.parametrize("question,expected", [
    ("what is sniffy nose", True), ("Sniffynest hosts in Pune", True), ("dog food under 1000", True),
    ("do you have cat toys", True), ("book grooming", True), ("how is nifty today", False),
    ("which category is this", False), ("who is the president", False),
])
def test_routing(question, expected):
    assert sniffy.is_about(question) is expected


@pytest.mark.parametrize("question,expected", [
    ("show me dog food", True), ("any cat toys under 500", True), ("I need a dog harness", True),
    ("can my dog eat chocolate food", False), ("how often should I buy dog food", False), ("what is sniffynose", False),
    ("book grooming", False), ("my cat is vomiting after food", False), ("food prices in India", False),
])
def test_shopping_requests(site, question, expected):
    assert sniffy.is_shopping(question) is expected


@pytest.mark.parametrize("question,expected", [
    ("show me rope toys", True), ("I want chicken food", True), ("show me food", False),
    ("find me a flight to Pune", False), ("show me a rocket toy", False), ("what is the news on rope toys", False),
])
def test_sold_here_even_without_naming_a_pet(site, question, expected):
    assert sniffy.is_shopping(question) is expected


def test_products_with_price_stock_and_picture(site):
    out = sniffy.search("show me dog food")
    names = [p["name"] for p in out["products"]]
    assert names[0] == "Chicken Dog Food" and "Puppy Dry Food" in names  # puppy counts as dog
    first = out["products"][0]
    assert first["price_inr"] == 1738 and first["in_stock"] and first["image"].startswith("https://")
    assert not any("/api/" in u for u in site)


def test_price_range_across_sizes():
    data = {"@type": "Product", "name": "Harness",
            "offers": {"@type": "AggregateOffer", "lowPrice": 3325, "highPrice": 4464.05,
                       "availability": "https://schema.org/InStock"}}
    page = f'<script type="application/ld+json">{json.dumps(data)}</script>'
    got = sniffy.parse_product(page, S + "/products/harness")
    assert (got["price_inr"], got["price_up_to_inr"], got["in_stock"]) == (3325, 4464.05, True)


def test_wrong_kind_of_product_is_not_offered(site):
    out = sniffy.search("do you have a cat bed")
    assert out["products"] == [] and "No matching product" in out["note"]
    assert not any("cat-litter" in u for u in site)


def test_service_question_shows_no_products(site):
    out = sniffy.search("grooming for my dog")
    assert out["products"] == [] and out["pages"][0]["title"] == "Grooming" and "note" not in out


def test_price_limit(site):
    out = sniffy.search("dog food under Rs 1,000")
    assert [p["name"] for p in out["products"]] == ["Puppy Dry Food"]  # toys are not food, 1738 is too dear
    assert out["price_limit_inr"] == 1000


def test_service_and_general_questions(site):
    assert sniffy.search("grooming")["pages"][0]["title"] == "Grooming"
    general = sniffy.search("what is sniffynose")
    assert [p["url"] for p in general["pages"]] == [S + "/", S + "/why-sniffynose"]


def test_sniffynest_is_linked_but_not_read(site):
    out = sniffy.search("sniffynest pet hosts")
    assert out["sniffynest"]["url"] == sniffy.NEST
    assert not any("nest.sniffynose.com" in u for u in site)


def test_fetch_refuses_sniffynest_unless_allowed():
    with pytest.raises(live.ToolError, match="asks automated tools"):
        live.fetch("https://nest.sniffynose.com/")


def test_product_cards_for_the_screen(site):
    session = live.Session("dog food")
    told = session.run("sniffynose", json.dumps({"query": "dog food"}))
    assert "cdn.example.com" not in told  # pictures aren't sent to the model
    card = session.cards[0]
    assert card["title"] == "Chicken Dog Food" and card["snippet"] == "₹1,738 · In stock" and card["image"]


def test_screen_gets_sources_and_compare_links(site):
    session = live.Session("dog food")
    told = session.run("sniffynose", json.dumps({"query": "SniffyNose dog food for pomeranian"}))
    assert "amazon" not in told.lower()  # links are for the screen, not for the model to read out
    assert session.sources and all(s["name"] == "sniffynose.com" for s in session.sources)
    assert [x["name"] for x in session.shops] == ["Amazon.in", "Google Shopping"]
    assert session.shops[0]["url"] == "https://www.amazon.in/s?k=dog+food+for+pomeranian"
    assert not any("amazon" in u or "google" in u for u in site)  # nothing is read from those shops


def test_list_answers_are_not_read_as_dashes():
    raw = "Here are some options: - Drools Wet Food, ₹700.\n- SuperCoat Dry Food.\n2. A third one."
    assert llm.clean_for_speech(raw) == "Here are some options: Drools Wet Food, ₹700. SuperCoat Dry Food. A third one."


def test_pet_questions_must_use_the_lookup_first(monkeypatch):
    sent = []

    def create(**kw):
        sent.append(kw)
        message = NS(content="Here you go.", tool_calls=None)
        return NS(choices=[NS(message=message)], usage=None)

    monkeypatch.setattr(llm, "_groq_client", lambda t: NS(chat=NS(completions=NS(create=create))))
    llm.answer([], "any dog food?", use_web=True)
    assert sent[-1]["tool_choice"]["function"]["name"] == "sniffynose"
    llm.answer([], "weather in Pune", use_web=True)
    assert sent[-1]["tool_choice"] == "auto"
    monkeypatch.setenv("SNIFFYNOSE", "0")
    llm.answer([], "any dog food?", use_web=True)
    assert sent[-1]["tool_choice"] == "auto"
