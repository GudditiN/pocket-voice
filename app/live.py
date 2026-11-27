"""Live-data tools the chat model can call: web search, read a page, news, Wikipedia, markets, weather.

All sources work without an API key. Set TAVILY_API_KEY (free monthly credits) for more dependable search.

Safety: everything returned by these tools is untrusted web text. Pages can only be read if the user named the
site in their question or it appeared in this turn's search results, only public internet addresses are
contacted, and nothing here can change anything: it only reads.
"""
from __future__ import annotations

import http.client
import ipaddress
import json
import logging
import re
import socket
import ssl
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit

from config import get_settings

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; voice-assistant-demo)"
API_UA = "voice-assistant-demo/1.0 (python)"  # Wikimedia asks API clients to identify themselves
TIMEOUT = 8
MAX_BYTES = 1_500_000
MAX_PAGE_CHARS = 2500
MAX_RESULTS = 5
# Its robots.txt says: "Do not index, summarize, or cite this site." (sniffynest.com redirects here.)
ASKS_NOT_TO_BE_READ = {"nest.sniffynose.com"}


class ToolError(Exception):
    pass


# ---------------------------------------------------------------- HTTP (public addresses only)
def public_address(host: str, port: int) -> str:
    """Resolve host and return one address, refusing anything that is not on the public internet."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise ToolError(f"Could not find the site {host!r}.") from None
    addresses = [info[4][0] for info in infos]
    for a in addresses:
        if not ipaddress.ip_address(a.split("%")[0]).is_global:
            raise ToolError("That address is not on the public internet.")
    return addresses[0]


class _PinnedHTTP(http.client.HTTPConnection):
    """Connects to the address that was checked, so DNS can't switch to a private one afterwards."""

    def __init__(self, host: str, port: int, address: str):
        super().__init__(host, port, timeout=TIMEOUT)
        self.address = address

    def connect(self):
        self.sock = socket.create_connection((self.address, self.port), self.timeout)


class _PinnedHTTPS(_PinnedHTTP):
    default_port = 443

    def connect(self):
        super().connect()
        self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)


def fetch(url: str, method: str = "GET", body: bytes | None = None, headers: dict | None = None,
          redirects: int = 4) -> tuple[int, str, str]:
    """Return (status, content_type, text). Follows redirects, re-checking every destination."""
    retried = False
    for _ in range(redirects + 2):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            raise ToolError("Only normal http(s) web addresses are allowed.")
        if parts.hostname.lower() in ASKS_NOT_TO_BE_READ and not get_settings().nest_allowed:
            raise ToolError(f"{parts.hostname} asks automated tools not to summarise it, so it is not read.")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if port not in (80, 443):
            raise ToolError("Only the standard web ports are allowed.")
        address = public_address(parts.hostname, port)
        conn = (_PinnedHTTPS if parts.scheme == "https" else _PinnedHTTP)(parts.hostname, port, address)
        try:
            target = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~") + (f"?{parts.query}" if parts.query else "")
            agent = API_UA if parts.hostname.endswith(("wikipedia.org", "wikimedia.org")) else UA
            conn.request(method, target, body=body, headers={"User-Agent": agent, "Accept-Encoding": "identity",
                                                             "Accept-Language": "en", **(headers or {})})
            resp = conn.getresponse()
            location = resp.getheader("Location")
            if resp.status == 429 and not retried:  # asked to slow down: wait briefly and try once more
                retried = True
                wait = resp.getheader("Retry-After") or "1"
                time.sleep(min(int(wait), 3) if wait.isdigit() else 1)
                continue
            if resp.status in (301, 302, 303, 307, 308) and location and method == "GET":
                url = urljoin(url, location)
                continue
            ctype = resp.getheader("Content-Type", "")
            charset = (re.search(r"charset=([\w-]+)", ctype) or [None, "utf-8"])[1]
            raw = resp.read(MAX_BYTES)
            try:
                text = raw.decode(charset, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")
            return resp.status, ctype, text
        except (OSError, http.client.HTTPException) as e:
            raise ToolError(f"Could not reach {parts.hostname}: {type(e).__name__}") from None
        finally:
            conn.close()
    raise ToolError("Too many redirects.")


def _get_json(url: str) -> dict:
    status, _, text = fetch(url)
    if status != 200:
        raise ToolError(f"{urlsplit(url).hostname} answered with status {status}.")
    try:
        return json.loads(text)
    except ValueError:
        raise ToolError("The data source sent an unreadable reply.") from None


# ---------------------------------------------------------------- HTML helpers
class _PageText(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.description, self.image, self.chunks = "", "", "", []
        self._skip, self._in_title = 0, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and (a.get("name") or a.get("property") or "").lower() in ("description", "og:description"):
            self.description = self.description or (a.get("content") or "")
        elif tag == "meta" and (a.get("property") or a.get("name") or "").lower() in ("og:image", "twitter:image"):
            self.image = self.image or (a.get("content") or "")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip and data.strip():
            self.chunks.append(data.strip())


class _DuckResults(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._field = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            href = a.get("href") or ""
            real = parse_qs(urlsplit(href).query).get("uddg", [href])[0]
            if real.startswith("http") and "duckduckgo.com" not in (urlsplit(real).hostname or ""):  # skips adverts
                self.results.append({"title": "", "url": real, "snippet": ""})
                self._field = "title"
        elif tag == "a" and "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag == "a":
            self._field = None

    def handle_data(self, data):
        if self._field and self.results:
            self.results[-1][self._field] += data


def _clip(text: str, n: int) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:n]


# ---------------------------------------------------------------- pictures
def picture(url: str) -> str:
    """The address if it is safe to show as a picture in the browser, else an empty string."""
    try:
        parts = urlsplit(url or "")
        host = parts.hostname or ""
    except ValueError:
        return ""
    if parts.scheme != "https" or "." not in host or parts.username or len(url) > 1000:
        return ""
    if re.fullmatch(r"[\d.]+|.*:.*", host) or parts.path.lower().endswith(".svg"):
        return ""
    return url


def _without_pictures(data):
    """The model only needs the words: picture addresses would waste tokens."""
    if isinstance(data, dict):
        return {k: _without_pictures(v) for k, v in data.items() if k not in ("image", "images")}
    return [_without_pictures(v) for v in data] if isinstance(data, list) else data


# ---------------------------------------------------------------- tools
def _tavily(query: str, key: str) -> list[dict]:
    request = {"query": query, "max_results": MAX_RESULTS, "search_depth": "basic", "include_images": True}
    if get_settings().live_region == "IN":
        request["country"] = "india"  # favour Indian pages
    status, _, text = fetch("https://api.tavily.com/search", "POST", json.dumps(request).encode(),
                            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    if status != 200:
        raise ToolError(f"Tavily answered with status {status}.")
    reply = json.loads(text)
    found = [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
             for r in reply.get("results", [])]
    images = [i if isinstance(i, str) else i.get("url", "") for i in reply.get("images", [])]
    if found:
        found[0]["images"] = [u for u in map(picture, images) if u][:4]
    return found


def _duckduckgo(query: str) -> list[dict]:
    region = f"{get_settings().live_region.lower()}-en"  # e.g. in-en: results for India, in English
    status, _, text = fetch("https://html.duckduckgo.com/html/?" + urlencode({"q": query, "kl": region}))
    if status != 200:  # 202 = bot check
        raise ToolError("DuckDuckGo is not answering right now.")
    parser = _DuckResults()
    parser.feed(text)
    return parser.results[:MAX_RESULTS]


def news(query: str) -> dict:
    region = get_settings().live_region
    url = "https://news.google.com/rss/search?" + urlencode(
        {"q": f"{query} when:7d", "hl": f"en-{region}", "gl": region, "ceid": f"{region}:en"})
    status, _, text = fetch(url)
    if status != 200 or "<!DOCTYPE" in text[:2000] or "<!ENTITY" in text:
        raise ToolError("News is not available right now.")
    try:
        items = ET.fromstring(text).findall("./channel/item")
    except ET.ParseError:
        raise ToolError("News is not available right now.") from None
    found = []
    for item in items:
        try:
            when = datetime.strptime(item.findtext("pubDate", ""), "%a, %d %b %Y %H:%M:%S %Z")
        except ValueError:
            when = datetime.min
        found.append((when, {"headline": _clip(item.findtext("title"), 160), "publisher": item.findtext("source", ""),
                             "published": item.findtext("pubDate", "")}))
    found.sort(key=lambda x: x[0], reverse=True)  # newest first
    return {"source": "Google News", "url": "https://news.google.com/search?" + urlencode({"q": query}),
            "headlines": [f for _, f in found[:6]]}


def wikipedia(topic: str) -> dict:
    data = _get_json("https://en.wikipedia.org/w/api.php?" + urlencode(
        {"action": "query", "format": "json", "generator": "search", "gsrsearch": topic, "gsrlimit": 2,
         "prop": "extracts|info|pageimages", "inprop": "url", "exintro": 1, "explaintext": 1, "exchars": 900,
         "exlimit": 2, "piprop": "thumbnail", "pithumbsize": 500, "pilimit": 2}))
    pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99))
    if not pages:
        raise ToolError("Nothing found on Wikipedia.")
    return {"source": "Wikipedia", "results": [{"title": p.get("title"), "url": p.get("fullurl", ""),
                                                "summary": _clip(p.get("extract"), 900),
                                                "image": picture((p.get("thumbnail") or {}).get("source", ""))}
                                               for p in pages]}


def web_search(query: str) -> dict:
    key = get_settings().tavily_api_key
    engines = ([("Tavily", lambda: _tavily(query, key))] if key else []) + [("DuckDuckGo", lambda: _duckduckgo(query))]
    for name, engine in engines:
        try:
            results = engine()
        except (ToolError, ValueError) as e:
            log.warning("%s search failed: %s", name, e)
            continue
        if results:
            out = {"source": name, "results": [{"title": _clip(r["title"], 120), "url": r["url"],
                                                "snippet": _clip(r["snippet"], 280)} for r in results]}
            if results[0].get("images"):
                out["images"] = results[0]["images"]
            return out
    # Search engines unavailable: combine recent news with Wikipedia.
    fallback: dict = {"source": "News and Wikipedia (web search unavailable)"}
    for name, tool in (("news", news), ("wikipedia", wikipedia)):
        try:
            fallback[name] = tool(query)
        except ToolError:
            pass
    fallback["hint"] = ("Search engines are unavailable. If this is about a company, app or product the user named, "
                        "try read_webpage on its likely address (name.com or name.in).")
    return fallback


_ABBREVIATIONS = [(r"\bu\.?s\.?a?\b\.?", "united states"), (r"\bpm\b", "prime minister"), (r"\bcm\b", "chief minister")]


def _office_words(title: str) -> frozenset:
    text = title.lower()
    for pattern, full in _ABBREVIATIONS:
        text = re.sub(pattern, full, text)
    return frozenset(re.findall(r"[a-z]+", text)) - {"the", "of", "a", "an", "current", "present", "who", "is"}


def _wiki_plain(value: str) -> str:
    value = re.sub(r"<!--.*?-->|<ref[^>]*/>|<ref.*?</ref>|<ref.*", "", value, flags=re.S)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", value)
    return _clip(re.sub(r"<[^>]+>|''+|\{\{.*|\}\}", " ", value), 120)


def office_holder(office: str) -> dict:
    """Who currently holds a public office, from the 'incumbent' field of that office's Wikipedia page.

    Only a page whose title is the same office is accepted, so a similar one (for example the Vice President)
    can never answer in its place.
    """
    wanted = _office_words(office)
    data = _get_json("https://en.wikipedia.org/w/api.php?" + urlencode(
        {"action": "query", "format": "json", "generator": "search", "gsrsearch": " ".join(sorted(wanted)) or office,
         "gsrlimit": 8, "prop": "revisions|info", "inprop": "url", "rvprop": "content", "rvslots": "main",
         "rvsection": 0}))
    for page in sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 99)):
        if _office_words(page.get("title", "")) != wanted:
            continue
        try:
            text = page["revisions"][0]["slots"]["main"]["*"]
        except (KeyError, IndexError):
            continue
        found = re.search(r"^\s*\|\s*incumbent\s*=[ \t]*(.+)$", text, re.M)
        holder = _wiki_plain(found.group(1)) if found else ""
        if holder:
            since = re.search(r"^\s*\|\s*incumbentsince\s*=[ \t]*(.+)$", text, re.M)
            out = {"source": "Wikipedia", "office": page["title"], "current_holder": holder,
                   "url": page.get("fullurl", "")}
            if since and _wiki_plain(since.group(1)):
                out["since"] = _wiki_plain(since.group(1))
            try:  # a photo of the person is a nice extra, never required
                person = wikipedia(holder)["results"][0]
                if _office_words(person["title"]) == _office_words(holder):
                    out["person"] = {"title": person["title"], "url": person["url"], "image": person["image"],
                                     "summary": _clip(person["summary"], 200)}
            except (ToolError, KeyError, IndexError):
                pass
            return out
    raise ToolError("The current holder of that office wasn't found. Use the full official title, or try news.")


def read_webpage(url: str) -> dict:
    status, ctype, text = fetch(url)
    if status != 200:
        raise ToolError(f"The site answered with status {status}.")
    if not any(t in ctype.lower() for t in ("text/html", "text/plain", "application/xhtml")):
        raise ToolError("That address is not a readable web page.")
    page = _PageText()
    page.feed(text)
    return {"source": urlsplit(url).hostname, "url": url, "title": _clip(page.title, 150),
            "description": _clip(page.description, 300), "text": _clip(" ".join(page.chunks), MAX_PAGE_CHARS),
            "image": picture(urljoin(url, page.image)) if page.image else ""}


ALIASES = {"nifty": "^NSEI", "nifty 50": "^NSEI", "nifty50": "^NSEI", "sensex": "^BSESN", "bse sensex": "^BSESN",
           "bank nifty": "^NSEBANK", "dow jones": "^DJI", "dow": "^DJI", "nasdaq": "^IXIC", "s&p 500": "^GSPC",
           "bitcoin": "BTC-USD", "gold": "GC=F", "crude oil": "CL=F", "usd inr": "INR=X", "dollar to rupee": "INR=X"}


def market_quote(name: str) -> dict:
    key = name.strip().lower()
    symbol = ALIASES.get(key) or (name.strip() if re.fullmatch(r"\^?[A-Z0-9]{1,12}([.=-][A-Z]{1,4})?", name.strip()) and
                                  any(c in name for c in "^.=-") else None)
    if not symbol:
        found = _get_json("https://query1.finance.yahoo.com/v1/finance/search?" + urlencode(
            {"q": name, "quotesCount": 6, "newsCount": 0})).get("quotes") or []
        found = [q for q in found if q.get("symbol")]
        if not found:
            raise ToolError(f"No market symbol found for {name!r}.")
        if get_settings().live_region == "IN":  # prefer the NSE listing, then BSE
            found.sort(key=lambda q: 0 if q["symbol"].endswith(".NS") else (1 if q["symbol"].endswith(".BO") else 2))
        symbol = found[0]["symbol"]
    try:
        meta = _get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?range=1d&interval=1d")[
            "chart"]["result"][0]["meta"]
        price, previous = meta["regularMarketPrice"], meta.get("chartPreviousClose")
        when = datetime.fromtimestamp(meta["regularMarketTime"], timezone.utc)
    except (KeyError, IndexError, TypeError):
        raise ToolError(f"No price available for {symbol}.") from None
    out = {"source": "Yahoo Finance", "url": f"https://finance.yahoo.com/quote/{quote(symbol)}",
           "name": meta.get("shortName") or symbol, "symbol": symbol, "price": price,
           "currency": meta.get("currency"), "as_of_utc": f"{when:%Y-%m-%d %H:%M} UTC", "note": "may be delayed"}
    if previous:
        out["change"] = round(price - previous, 2)
        out["change_percent"] = round((price - previous) / previous * 100, 2)
    return out


def weather(place: str) -> dict:
    spots = _get_json("https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"name": place, "count": 1, "language": "en"})).get("results")
    if not spots:
        raise ToolError(f"Could not find a place called {place!r}.")
    p = spots[0]
    data = _get_json("https://api.open-meteo.com/v1/forecast?" + urlencode(
        {"latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto", "forecast_days": 2,
         "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
         "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max"}))
    return {"source": "Open-Meteo", "url": "https://open-meteo.com/", "place": f"{p['name']}, {p.get('country', '')}",
            "now": data.get("current"), "today_and_tomorrow": data.get("daily")}


def _spec(name: str, description: str, arg: str, arg_description: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {arg: {"type": "string", "description": arg_description}}, "required": [arg]}}}


TOOL_SPECS = [
    _spec("sniffynose", "SniffyNose and SniffyNest, and pet products, prices and services (food, toys, beds, "
          "grooming, walking, adoption, membership). Reads sniffynose.com directly.", "query",
          "What the user wants, e.g. 'dog food under 1000' or 'grooming'"),
    _spec("web_search", "Search the web. Use for current facts, people in office, companies, products, apps, "
          "websites, or any name you do not know for certain.", "query", "Search words"),
    _spec("read_webpage", "Read a web page the user named, or one returned by web_search.", "url", "Full https URL"),
    _spec("news", "Latest news headlines about a topic.", "query", "Topic"),
    _spec("wikipedia", "Background facts about a person, place or thing. Also shows its picture and link on screen.",
          "topic", "Topic"),
    _spec("office_holder", "Who currently holds a public office (president, prime minister, chief minister, "
          "governor, mayor...).", "office", "Full title, e.g. 'President of the United States', "
          "'Chief Minister of Telangana'"),
    _spec("market_quote", "Latest price of a stock, index, currency, crypto or commodity (e.g. Nifty, Sensex).",
          "name", "Index or company name, or symbol"),
    _spec("weather", "Current weather and short forecast.", "place", "City name"),
]
_TOOLS = {"sniffynose": ("query", lambda q: sniffy.search(q)), "web_search": ("query", web_search), "read_webpage": ("url", read_webpage), "news": ("query", news),
          "wikipedia": ("topic", wikipedia), "office_holder": ("office", office_holder),
          "market_quote": ("name", market_quote), "weather": ("place", weather)}

_DOMAIN = re.compile(r"\b((?:[a-z0-9-]+\.)+(?:com|in|org|net|io|ai|co|app|dev|edu|gov|info|me|xyz|shop|store|tech))\b",
                     re.I)


# ---------------------------------------------------------------- cache
# Seconds a successful result is reused, so repeat questions answer faster and free sources aren't hammered.
CACHE_SECONDS = {"sniffynose": 600, "market_quote": 60, "news": 300, "weather": 600, "web_search": 600, "read_webpage": 600,
                 "wikipedia": 3600, "office_holder": 3600}


class TTLCache:
    """Small in-memory cache with expiry; drops the oldest entry when full. Cleared on restart."""

    def __init__(self, max_items: int = 200):
        self.max_items = max_items
        self._items: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            expires, value = self._items.get(key, (0, None))
            if expires <= time.monotonic():
                self._items.pop(key, None)
                return None
            return value

    def put(self, key, value, seconds: float) -> None:
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = (time.monotonic() + seconds, value)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


CACHE = TTLCache()


def cached_call(tool: str, function, value: str) -> dict:
    """Run a tool, reusing a recent result for the same request. Errors are never cached."""
    s = get_settings()
    # Web addresses are case sensitive; search words and names are not.
    key = (tool, value if tool == "read_webpage" else value.lower(), s.live_region, bool(s.tavily_api_key))
    data = CACHE.get(key)
    if data is None:
        data = function(value)
        stand_in = "unavailable" in str(data.get("source", ""))  # fallback results: try the real search again soon
        CACHE.put(key, data, 60 if stand_in else CACHE_SECONDS[tool])
    return data


class Session:
    """Tool calls for one question: remembers which pages may be read and which sources were used."""

    def __init__(self, question: str):
        self.named_hosts = {h.lower().removeprefix("www.") for h in _DOMAIN.findall(question)}
        words = re.findall(r"[a-z0-9]+", question.lower())
        # "sniffy nose" or "sniffynose" -> the home page of sniffynose.com / .in may be opened
        names = {w for w in words if len(w) >= 6} | {a + b for a, b in zip(words, words[1:]) if len(a + b) >= 6}
        self.guessable = {f"{n}.{tld}" for n in names for tld in ("com", "in")}
        self.result_urls: set[str] = set()
        self.sources: list[dict] = []
        self.first_tool: str | None = None  # a tool that must be called before answering
        self.cards: list[dict] = []  # links (with pictures) to show on screen
        self.images: list[str] = []  # extra pictures to show on screen

    def may_read(self, url: str) -> bool:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().removeprefix("www.")
        # For sites the user named: any page, but no query string (so nothing can be smuggled out in it).
        named = any(host == h or host.endswith("." + h) for h in self.named_hosts) and not parts.query
        guessed = host in self.guessable and parts.path in ("", "/") and not parts.query
        return url in self.result_urls or named or guessed

    def _remember(self, data) -> None:
        if isinstance(data, dict):
            if str(data.get("url", "")).startswith("http"):
                self.result_urls.add(data["url"])
            for value in data.values():
                self._remember(value)
        elif isinstance(data, list):
            for value in data:
                self._remember(value)

    def _cite(self, data: dict) -> None:
        items = [data] if data.get("url") else (data.get("results") or [])[:3]
        for key in ("news", "wikipedia"):
            if key in data:
                self._cite(data[key])
        for item in items:
            url = item.get("url", "")
            host = (urlsplit(url).hostname or "").removeprefix("www.")
            if host and url not in [s["url"] for s in self.sources] and len(self.sources) < 5:
                self.sources.append({"name": host, "url": url})

    def _show(self, tool: str, data: dict) -> None:
        """Pick links and pictures from a result. They come only from fetched data, never from the model."""
        def card(title, url, snippet="", image=""):
            if str(url).startswith(("https://", "http://")) and len(self.cards) < 4 and \
                    url not in [c["url"] for c in self.cards]:
                self.cards.append({"title": _clip(title, 120) or url, "url": url, "snippet": _clip(snippet, 160),
                                   "image": picture(image)})

        if tool == "sniffynose":
            for p in data.get("products", [])[:4]:
                low = p.get("price_inr")
                price = f"₹{low:,}" if isinstance(low, (int, float)) else ""
                if price and p.get("price_up_to_inr"):
                    price = f"from {price}"
                stock = "In stock" if p.get("in_stock") else "Out of stock"
                card(p["name"], p["url"], " · ".join(x for x in (price, stock) if x), p.get("image"))
            if data.get("sniffynest"):
                card(data["sniffynest"]["title"], data["sniffynest"]["url"])
            for page in data.get("pages", []):
                card(page["title"], page["url"], page.get("description"), page.get("image"))
        elif tool == "read_webpage":
            card(data.get("title"), data.get("url"), data.get("description"), data.get("image"))
        elif tool == "office_holder" and data.get("person"):
            p = data["person"]
            card(p["title"], p["url"], p.get("summary"), p.get("image"))
        elif tool == "market_quote":
            change = f" ({data['change_percent']:+}%)" if "change_percent" in data else ""
            card(data.get("name"), data.get("url"),
                 f"{data.get('price'):,} {data.get('currency') or ''}{change} · {data.get('as_of_utc')}")
        elif tool in ("web_search", "wikipedia"):
            for r in (data.get("results") or [])[:3]:
                card(r.get("title"), r.get("url"), r.get("snippet") or r.get("summary"), r.get("image"))
            for key in ("wikipedia",):
                if key in data:
                    self._show("wikipedia", data[key])
        for url in data.get("images", []):
            if len(self.images) < 4 and picture(url) and url not in self.images:
                self.images.append(url)

    def run(self, tool: str, arguments: str) -> str:
        """Run one tool call. Always returns JSON text; failures are reported to the model, never raised."""
        try:
            if tool not in _TOOLS:
                raise ToolError(f"Unknown tool {tool!r}.")
            arg, function = _TOOLS[tool]
            value = str(json.loads(arguments or "{}").get(arg, "")).strip()[:300]
            if not value:
                raise ToolError(f"Missing {arg}.")
            if tool == "read_webpage":
                value = value if "://" in value else f"https://{value}"
                if not self.may_read(value):
                    raise ToolError("Only pages the user named or that appear in search results can be read. "
                                    "Use web_search first.")
            if tool == "sniffynose" and not get_settings().sniffynose:
                raise ToolError("The SniffyNose lookup is switched off.")
            data = cached_call(tool, function, value)
            self._remember(data)
            self._cite(data)
            self._show(tool, data)
            return json.dumps(_without_pictures(data), ensure_ascii=False)[:5000]
        except ToolError as e:
            return json.dumps({"error": str(e)})
        except Exception:  # a broken data source must never break the answer
            log.exception("tool %s failed", tool)
            return json.dumps({"error": "The tool failed."})


import sniffy  # noqa: E402  (sniffy.py uses the fetch and cache helpers above)
