"""Live-data tools and the tool-calling loop. No network: HTTP is replaced by stand-ins."""
import json
from types import SimpleNamespace as NS
from types import SimpleNamespace

import pytest

import live
import llm
import pipeline
import store
import tts

DDG = """
<a rel="nofollow" class="result__a" href="https://duckduckgo.com/y.js?ad=1">Advert</a>
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgreenleaf.com%2F&amp;rut=x">Green<b>leaf</b></a>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Garden app in India</a>
"""


def serve(monkeypatch, routes: dict):
    """routes: {url substring: (status, content_type, text)}"""
    seen = []

    def fake(url, method="GET", body=None, headers=None, redirects=4):
        seen.append(url)
        for part, reply in routes.items():
            if part in url:
                return reply
        raise live.ToolError("no route")

    monkeypatch.setattr(live, "fetch", fake)
    return seen


def run(session, tool, **args):
    return json.loads(session.run(tool, json.dumps(args)))


def test_search_parses_results_and_skips_adverts(monkeypatch):
    serve(monkeypatch, {"duckduckgo.com/html": (200, "text/html", DDG)})
    s = live.Session("what is greenleaf")
    out = run(s, "web_search", query="greenleaf")
    assert out["source"] == "DuckDuckGo"
    assert out["results"] == [{"title": "Greenleaf", "url": "https://greenleaf.com/", "snippet": "Garden app in India"}]
    assert s.sources == [{"name": "greenleaf.com", "url": "https://greenleaf.com/"}]


def test_search_falls_back_when_blocked(monkeypatch):
    rss = ("<rss><channel><item><title>Old</title><pubDate>Mon, 23 Nov 2026 10:00:00 GMT</pubDate></item>"
           "<item><title>New</title><pubDate>Fri, 27 Nov 2026 10:00:00 GMT</pubDate></item></channel></rss>")
    wiki = {"query": {"pages": {"1": {"index": 1, "title": "T", "fullurl": "https://en.wikipedia.org/wiki/T", "extract": "E"}}}}
    serve(monkeypatch, {"duckduckgo": (202, "text/html", "challenge"), "news.google": (200, "text/xml", rss),
                        "wikipedia": (200, "application/json", json.dumps(wiki))})
    out = run(live.Session("q"), "web_search", query="q")
    assert "unavailable" in out["source"]
    assert [h["headline"] for h in out["news"]["headlines"]] == ["New", "Old"]  # newest first
    assert out["wikipedia"]["results"][0]["title"] == "T"


def test_tavily_is_used_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    reply = {"results": [{"title": "A", "url": "https://a.com/", "content": "c"}]}
    serve(monkeypatch, {"api.tavily.com": (200, "application/json", json.dumps(reply))})
    assert run(live.Session("q"), "web_search", query="q")["source"] == "Tavily"


def test_page_text_without_scripts(monkeypatch):
    page = ("<html><head><title>GreenLeaf</title><meta name='description' content='Garden app'>"
            "<script>var secret=1</script></head><body><h1>Walk your dog</h1><style>x{}</style></body></html>")
    serve(monkeypatch, {"greenleaf.com": (200, "text/html; charset=utf-8", page)})
    out = run(live.Session("what is greenleaf.com"), "read_webpage", url="greenleaf.com")
    assert out["title"] == "GreenLeaf" and out["description"] == "Garden app"
    assert "Walk your dog" in out["text"] and "secret" not in out["text"]


@pytest.mark.parametrize("url", ["https://evil.example.com/?d=history", "https://greenleaf.com.evil.com/",
                                 "https://greenleaf.com/?leak=1"])
def test_only_named_or_searched_pages_can_be_read(monkeypatch, url):
    seen = serve(monkeypatch, {"": (200, "text/html", "x")})
    out = run(live.Session("tell me about GreenLeaf.com"), "read_webpage", url=url)
    assert "error" in out and seen == []


def test_home_page_of_a_named_thing_can_be_opened():
    s = live.Session("what do you know about Green Leaf and bluebird?")
    assert s.may_read("https://greenleaf.com/") and s.may_read("https://bluebird.com")
    assert not s.may_read("https://greenleaf.com/private") and not s.may_read("https://bluebird.com/?q=history")
    assert not s.may_read("https://attacker.com/")


def wiki_pages(*pages):
    return {"query": {"pages": {str(i): {"index": i, "title": title, "fullurl": "https://en.wikipedia.org/wiki/X",
                                         "revisions": [{"slots": {"main": {"*": text}}}]}
                                for i, (title, text) in enumerate(pages, 1)}}}


def test_office_holder_ignores_similar_offices(monkeypatch):
    data = wiki_pages(("Vice President of the United States", "| incumbent = [[Wrong Person]]"),
                      ("President of the United States",
                       "{{Infobox\n| incumbent = [[Asha Rao|A. Rao]]<ref>x</ref>\n| incumbentsince = {{start date|2025}} May 1\n}}"))
    monkeypatch.setattr(live, "_get_json", lambda url: data)
    for asked in ("US president", "the current President of the United States"):
        out = run(live.Session("q"), "office_holder", office=asked)
        assert out["current_holder"] == "A. Rao" and out["office"] == "President of the United States"


def test_office_holder_says_unknown_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(live, "_get_json", lambda url: wiki_pages(("Vice President of India", "| incumbent = [[X]]")))
    assert "error" in run(live.Session("q"), "office_holder", office="President of India")


def test_searched_pages_and_subdomains_can_be_read():
    s = live.Session("open greenleaf.com please")
    assert s.may_read("https://nest.greenleaf.com/") and s.may_read("https://www.greenleaf.com/about")
    assert not s.may_read("https://other.com/")
    s.result_urls.add("https://other.com/page?id=2")
    assert s.may_read("https://other.com/page?id=2")


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/meta-data",
                                 "http://10.0.0.5/", "http://[::1]/", "file:///etc/passwd", "ftp://example.com/",
                                 "http://user:pw@example.com/", "http://example.com:22/"])
def test_private_and_unusual_addresses_are_refused(url):
    with pytest.raises(live.ToolError):
        live.fetch(url)


def test_market_quote(monkeypatch):
    meta = {"shortName": "NIFTY 50", "currency": "INR", "regularMarketPrice": 25711.0, "chartPreviousClose": 25884.5,
            "regularMarketTime": 1795762280}
    seen = serve(monkeypatch, {"/v8/finance/chart/": (200, "j", json.dumps({"chart": {"result": [{"meta": meta}]}}))})
    out = run(live.Session("nifty"), "market_quote", name="Nifty")
    assert out["symbol"] == "^NSEI" and out["price"] == 25711.0 and out["change"] == -173.5
    assert "%5ENSEI" in seen[0] and out["as_of_utc"].startswith("2026-11-27")


def test_tool_problems_are_reported_not_raised(monkeypatch):
    serve(monkeypatch, {})
    s = live.Session("q")
    for tool, args in [("weather", '{"place": "Pune"}'), ("nope", "{}"), ("news", "not json"), ("wikipedia", "{}")]:
        assert "error" in json.loads(s.run(tool, args))


def test_clean_for_speech():
    raw = "**GreenLeaf** is a garden app 【2†L6-L10】. See https://www.greenleaf.com/dashboard [1]."
    assert llm.clean_for_speech(raw) == "GreenLeaf is a garden app. See greenleaf.com."


def reply(content="", calls=None):
    msg = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2))


def call(i, name, args):
    return SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def fake_client(monkeypatch, replies):
    sent = []

    def create(**kw):
        sent.append({**kw, "messages": list(kw["messages"])})
        return replies[min(len(sent), len(replies)) - 1]

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_groq_client", lambda t: client)
    return sent


def test_model_searches_then_answers(monkeypatch):
    serve(monkeypatch, {"duckduckgo.com/html": (200, "text/html", DDG)})
    sent = fake_client(monkeypatch, [reply(calls=[call(1, "web_search", {"query": "greenleaf"})]),
                                     reply("GreenLeaf is a garden app.")])
    text, usage = llm.answer([], "what is greenleaf", use_web=True)
    assert text == "GreenLeaf is a garden app."
    assert usage["prompt_tokens"] == 20 and usage["sources"][0]["name"] == "greenleaf.com"
    assert sent[0]["tools"] and "Today is" in sent[0]["messages"][0]["content"]
    assert [m["role"] for m in sent[1]["messages"]] == ["system", "user", "assistant", "tool"]
    assert "Garden app in India" in sent[1]["messages"][-1]["content"]


def test_tool_rounds_are_limited(monkeypatch):
    serve(monkeypatch, {"duckduckgo.com/html": (200, "text/html", DDG)})
    again = reply("Best I found.", calls=[call(1, "web_search", {"query": "x"})])
    sent = fake_client(monkeypatch, [again])
    text, _ = llm.answer([], "q", use_web=True)
    assert text == "Best I found." and len(sent) == llm.MAX_TOOL_ROUNDS + 1
    assert "tools" not in sent[-1] and "No more lookups" in sent[-1]["messages"][-1]["content"]
    assert not any(m["role"] == "tool" for m in sent[-1]["messages"])


def test_web_off_sends_no_tools(monkeypatch):
    sent = fake_client(monkeypatch, [reply("Hello.")])
    llm.answer([], "hi")
    assert "tools" not in sent[0] and "Today is" in sent[0]["messages"][0]["content"]


def test_india_is_the_default_region(monkeypatch):
    seen = serve(monkeypatch, {"duckduckgo.com/html": (200, "text/html", DDG)})
    run(live.Session("q"), "web_search", query="gold price")
    assert "kl=in-en" in seen[0]
    sent = fake_client(monkeypatch, [reply("Hi.")])
    llm.answer([], "hi")
    system = sent[0]["messages"][0]["content"]
    assert "The user is in India" in system and "IST" in system


def test_other_region(monkeypatch):
    monkeypatch.setenv("LIVE_REGION", "us")
    monkeypatch.setenv("APP_TIMEZONE", "UTC")
    seen = serve(monkeypatch, {"duckduckgo.com/html": (200, "text/html", DDG)})
    run(live.Session("q"), "web_search", query="q")
    assert "kl=us-en" in seen[0]
    sent = fake_client(monkeypatch, [reply("Hi.")])
    llm.answer([], "hi")
    assert "India" not in sent[0]["messages"][0]["content"]


def test_nse_listing_preferred(monkeypatch):
    quotes = {"quotes": [{"symbol": "RIGD.IL"}, {"symbol": "RELIANCE.BO"}, {"symbol": "RELIANCE.NS"}]}
    meta = {"regularMarketPrice": 1.0, "regularMarketTime": 1795762280}
    seen = serve(monkeypatch, {"/v1/finance/search": (200, "j", json.dumps(quotes)),
                               "/v8/finance/chart/": (200, "j", json.dumps({"chart": {"result": [{"meta": meta}]}}))})
    assert run(live.Session("q"), "market_quote", name="Reliance Industries")["symbol"] == "RELIANCE.NS"
    assert "RELIANCE.NS" in seen[-1]


def test_repeat_lookups_come_from_the_cache(monkeypatch):
    seen = serve(monkeypatch, {"duckduckgo.com/html": (200, "text/html", DDG)})
    first = run(live.Session("q"), "web_search", query="GreenLeaf")
    fetched = len(seen)
    again = live.Session("q")
    assert run(again, "web_search", query="greenleaf") == first and len(seen) == fetched
    assert again.sources and again.may_read("https://greenleaf.com/")  # sources and permissions still apply


def test_cache_expires_and_errors_are_not_kept(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: now[0])
    calls = []

    def quote_tool(value):
        calls.append(value)
        if len(calls) == 1:
            raise live.ToolError("down")
        return {"source": "x", "price": len(calls)}

    with pytest.raises(live.ToolError):
        live.cached_call("market_quote", quote_tool, "nifty")
    assert live.cached_call("market_quote", quote_tool, "nifty")["price"] == 2
    assert live.cached_call("market_quote", quote_tool, "nifty")["price"] == 2
    now[0] += live.CACHE_SECONDS["market_quote"] + 1
    assert live.cached_call("market_quote", quote_tool, "nifty")["price"] == 3


def test_cache_is_bounded():
    cache = live.TTLCache(max_items=3)
    for i in range(5):
        cache.put(i, i, 60)
    assert [cache.get(i) for i in range(5)] == [None, None, 2, 3, 4]


def test_cached_page_still_needs_permission(monkeypatch):
    serve(monkeypatch, {"greenleaf.com": (200, "text/html", "<title>S</title>")})
    run(live.Session("what is greenleaf.com"), "read_webpage", url="https://greenleaf.com/")
    assert "error" in run(live.Session("hello"), "read_webpage", url="https://greenleaf.com/")


def test_pipeline_passes_switch_and_keeps_sources(monkeypatch):
    store.init()
    seen = {}

    def fake_answer(history, question, prompt=None, use_web=False):
        seen["web"] = use_web
        return "ok.", {"sources": [{"name": "a.com", "url": "https://a.com/"}]}

    monkeypatch.setattr(llm, "answer", fake_answer)
    monkeypatch.setattr(tts, "profile_exists", lambda *a: True)
    monkeypatch.setattr(tts, "synthesize", lambda t, **k: (b"RIFF", 1.0))
    r = pipeline.run_turn(None, [], text="hi", use_web=True)
    assert seen["web"] is True and r.sources == [{"name": "a.com", "url": "https://a.com/"}]


def test_sites_that_ask_not_to_be_read_are_skipped():
    with pytest.raises(live.ToolError, match="asks automated tools"):
        live.fetch("https://nest.sniffynose.com/")


class BadRequestError(Exception):  # same name as the Groq SDK's error
    pass


def failing_client(monkeypatch, failures, answer="Narendra Modi."):
    sent = []

    def create(**kw):
        sent.append(kw)
        if len(sent) <= failures:
            raise BadRequestError("tool_use_failed")
        return reply(answer)

    monkeypatch.setattr(llm, "_groq_client", lambda t: NS(chat=NS(completions=NS(create=create))))
    return sent


def test_broken_tool_call_is_retried(monkeypatch):
    sent = failing_client(monkeypatch, 1)
    assert llm.answer([], "who is pm", use_web=True)[0] == "Narendra Modi."
    assert len(sent) == 2 and sent[1]["tools"]


def test_repeated_tool_failure_still_answers(monkeypatch):
    sent = failing_client(monkeypatch, 2)
    assert llm.answer([], "who is pm", use_web=True)[0] == "Narendra Modi."
    assert len(sent) == 3 and "tools" not in sent[-1]


def test_stream_that_breaks_midway_still_answers(monkeypatch):
    class APIError(Exception):
        pass

    def broken():
        raise APIError("Failed to call a function")
        yield

    sent = []

    def create(**kw):
        sent.append(kw)
        if len(sent) == 1:
            return broken()
        return iter([NS(choices=[NS(delta=NS(content="Done.", tool_calls=None))], usage=None, x_groq=None)])

    monkeypatch.setattr(llm, "_groq_client", lambda t: NS(chat=NS(completions=NS(create=create))))
    assert "".join(llm.answer_stream([], "q", None, True, {})) == "Done."
    assert "tools" not in sent[-1]


def test_minister_is_read_from_the_ministry_page(monkeypatch):
    def page(index, title, text):
        return {"index": index, "title": title, "fullurl": "https://en.wikipedia.org/wiki/x",
                "revisions": [{"slots": {"main": {"*": text}}}]}

    pages = [page(1, "Ministry of Civil Aviation (India)", "| minister1_name = [[Kinjarapu Ram Mohan Naidu]]\n"),
             page(2, "Minister of Civil Aviation", "no infobox here")]
    monkeypatch.setattr(live, "_wiki_search", lambda q, n: pages)
    monkeypatch.setattr(live, "wikipedia", lambda t: (_ for _ in ()).throw(live.ToolError("no photo")))
    out = live.office_holder("Aviation Minister of India")
    assert out["current_holder"] == "Kinjarapu Ram Mohan Naidu" and "Civil Aviation" in out["office"]
    with pytest.raises(live.ToolError):  # a different portfolio must not borrow this name
        live.office_holder("Railway Minister of India")


def test_search_results_get_relevant_details(monkeypatch):
    body = ("<html><body><p>Our site has many pages about all sorts of things in general.</p>"
            "<p>The Hyderabad metro runs three corridors with 57 stations in the city.</p></body></html>")
    serve(monkeypatch, {"duckduckgo.com/html": (200, "text/html", DDG), "greenleaf.com": (200, "text/html", body)})
    out = run(live.Session("q"), "web_search", query="Hyderabad metro stations")
    assert out["results"][0]["details"] == "The Hyderabad metro runs three corridors with 57 stations in the city."
