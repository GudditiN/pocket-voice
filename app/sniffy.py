"""Answers about SniffyNose and pet products or services, read directly from the public sniffynose.com pages.

How the site's robots.txt is respected (checked 27 Nov 2026):
  * content signals "search=yes, ai-input=yes, ai-train=no, use=reference": pages may inform an answer that
    links back to them; they must not be used to train a model
  * "Disallow: /api/": only normal public pages and the published sitemaps are requested
  * nest.sniffynose.com (SniffyNest) asks automated tools not to summarise it, so it is only linked to, unless
    the owner allows it with SNIFFYNEST_ALLOWED=1
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import live
from config import get_settings

SITE = "https://sniffynose.com"
NEST = "https://nest.sniffynose.com/"
SITEMAPS = ("products", "categories", "brands", "static")
MAX_PRODUCTS = 6

_BRAND = re.compile(r"sniff(?:y|ie|i|ey)?[\s-]*(?:nose|nest|noes|nos\b)|sniffy", re.I)
_PET = re.compile(r"\b(?:pets?|dogs?|doggy|puppy|puppies|cats?|kittens?|kitty|groom(?:ing)?|dog walk(?:er|ing)?|"
                  r"pet (?:food|toys?|beds?|adoption|hosts?|boarding)|treats for|litter|leash|harness|kibble)\b", re.I)

_IGNORE = set("a an the is are am do does you your have has any me my i we show tell about what which where how much "
              "price prices cost of for in on at to and or with from give need want buy get best good please can "
              "sniffy sniffynose sniffynest nose nest product products item items pet pets under below less than rs "
              "rupees inr within there some my it can could book looking show list all find order shop sell search suggest "
              "recommend got".split())
_SPECIES = {"dog", "cat"}
_SERVICES = {"grooming", "walker", "adoption", "membership", "offer", "reward", "support", "location", "space", "vet"}
_SAME = {"puppy": "dog", "pup": "dog", "doggy": "dog", "kitten": "cat", "kitty": "cat", "walking": "walker",
         "walk": "walker", "adopt": "adoption", "adopting": "adoption", "member": "membership", "offer": "offer",
         "discount": "offer", "groom": "grooming", "kibble": "food", "clothes": "dress", "cloth": "dress"}


_ITEMS = re.compile(r"\b(?:products?|items?|food|treats?|toys?|beds?|harness(?:es)?|leash(?:es)?|collars?|litter|"
                    r"shampoos?|bowls?|feeders?|cages?|dress(?:es)?|clothes|carriers?|crates?|brush(?:es)?|"
                    r"supplements?|chews?|bones?|kibble|biscuits?|accessories|grooming kit)\b", re.I)
_ADVICE = re.compile(r"\b(?:can (?:my |a )?(?:dog|cat|pet|puppy|kitten)s?|should|why|safe|sick|ill|vomit\w*|diarr\w*|"
                     r"how (?:often|to|do i|can i|much (?:food )?should)|is it ok(?:ay)?|allerg\w*|vet|news)\b", re.I)
_BUY = re.compile(r"\b(?:show|buy|order|shop|need|want|find|get|looking for|search|any|suggest|recommend|"
                  r"do you (?:have|sell)|have you got)\b", re.I)
_EVERYDAY = {"food", "bed", "toy", "bowl", "brush", "dress", "carrier", "accessorie", "supplement"}  # also non-pet


def is_shopping(question: str) -> bool:
    """A request to see products (not advice or a pet-care question): answered with cards, no long reply."""
    if _ADVICE.search(question):
        return False
    if is_about(question) and _ITEMS.search(question):
        return True
    return bool(_BUY.search(question)) and in_catalogue(question)


def _wanted(query: str) -> set[str]:
    """The words of a request that describe the thing asked for."""
    return _words(query) - {_SAME.get(w, w) for w in _IGNORE} - _IGNORE


def _slug_words(url: str) -> set[str]:
    return _words(url.rstrip("/").rsplit("/", 1)[-1])


def in_catalogue(question: str) -> bool:
    """Does SniffyNose sell what was asked for, even if no pet is mentioned ("show me chew bones")?

    It must name a kind of product or a brand, and one product must match every word. A lone everyday word
    such as "food" is not enough, as it may well mean something else.
    """
    wanted = _wanted(question)
    if not wanted or wanted <= _EVERYDAY:
        return False
    try:
        if not (_ITEMS.search(question) or any(wanted & _slug_words(u) for u in _sitemap("brands"))):
            return False
        return any(wanted <= _slug_words(u) for u in _sitemap("products"))
    except live.ToolError:
        return False


def warm_up() -> None:
    """Fetch the product and page lists ahead of the first question."""
    for kind in SITEMAPS:
        try:
            _sitemap(kind)
        except live.ToolError:
            return


def is_about(question: str) -> bool:
    """Should this question be answered from SniffyNose first?"""
    return bool(_BRAND.search(question) or _PET.search(question))


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        w = _SAME.get(w, w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = _SAME.get(w[:-1], w[:-1])
        out.add(w)
    return out


def _price_limit(query: str) -> float | None:
    m = re.search(r"(?:under|below|less than|within|upto|up to)\s*(?:rs\.?|inr|₹)?\s*([\d,]{2,7})", query, re.I)
    return float(m.group(1).replace(",", "")) if m else None


def compare_links(query: str) -> list[dict]:
    """Ready-made searches on other shops. These are links only: nothing is read from those sites."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query)
             if not _BRAND.fullmatch(w) and w.lower() not in _IGNORE - {"for", "pet"} and w.lower() != "sniffy"]
    q = quote_plus(" ".join(words) or query)
    return [{"name": "Amazon.in", "url": f"https://www.amazon.in/s?k={q}"},
            {"name": "Google Shopping", "url": f"https://www.google.com/search?tbm=shop&q={q}"}]


def _get(url: str) -> str:
    status, _, text = live.fetch(url)
    if status != 200:
        raise live.ToolError(f"SniffyNose answered with status {status}.")
    return text


def _sitemap(kind: str) -> list[str]:
    key = ("sniffy-sitemap", kind)
    urls = live.CACHE.get(key)
    if urls is None:
        found = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", _get(f"{SITE}/sitemap-{kind}.xml"))
        urls = [u for u in found if u.startswith(SITE) and "/api/" not in u]
        live.CACHE.put(key, urls, 3600)
    return urls


def _best(urls: list[str], wanted: set[str], limit: int) -> list[str]:
    """Addresses whose last part shares the most words with the request.

    If the request names a kind of thing (bed, harness, food), that word must match: "cat bed" never
    returns cat food just because both mention cats.
    """
    kind = wanted - _SPECIES
    scored = []
    for url in urls:
        slug = _slug_words(url)
        score = len(wanted & slug)
        if score and (not kind or kind & slug):
            scored.append((-score, len(slug), url))
    return [url for _, _, url in sorted(scored)[:limit]]


def parse_product(html_text: str, url: str) -> dict | None:
    """The product details a page publishes for search engines (JSON-LD)."""
    for block in re.findall(r"<script[^>]+application/ld\+json[^>]*>(.*?)</script>", html_text, re.S):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        for item in data if isinstance(data, list) else data.get("@graph", [data]):
            if not isinstance(item, dict) or item.get("@type") != "Product":
                continue
            offers = item.get("offers") or {}
            offer = offers[0] if isinstance(offers, list) and offers else offers
            image = item.get("image")
            image = image[0] if isinstance(image, list) and image else image
            page = live._PageText()
            page.feed(html_text)
            # one price, or a range across sizes and colours ("AggregateOffer")
            low, high = offer.get("price", offer.get("lowPrice")), offer.get("highPrice")
            return {"name": live._clip(item.get("name"), 120), "url": url, "price_inr": low,
                    "price_up_to_inr": high if high != low else None,
                    "in_stock": str(offer.get("availability", "")).endswith("InStock"),
                    "description": live._clip(item.get("description") or page.description, 180),
                    "image": live.picture(image if isinstance(image, str) else "")}
    return None


def _read_product(url: str) -> dict | None:
    try:
        return parse_product(_get(url), url)
    except live.ToolError:
        return None


def _read_page(url: str) -> dict | None:
    try:
        page = live._PageText()
        page.feed(_get(url))
    except live.ToolError:
        return None
    return {"title": live._clip(page.title, 120), "url": url, "description": live._clip(page.description, 250),
            "text": live._clip(" ".join(page.chunks), 700), "image": live.picture(page.image)}


def search(query: str) -> dict:
    wanted = _wanted(query)
    limit = _price_limit(query)
    out: dict = {"source": "SniffyNose (sniffynose.com)", "products": [], "pages": []}

    things = wanted - _SERVICES - _SPECIES
    services_only = bool(wanted & _SERVICES) and not things  # "grooming for my dog" is not a shopping question
    product_urls = [] if services_only or not wanted else \
        _best(_sitemap("products"), wanted - _SERVICES, 14 if limit else MAX_PRODUCTS)
    with ThreadPoolExecutor(max_workers=6) as pool:  # a few pages at a time, to be gentle on the site
        products = [p for p in pool.map(_read_product, product_urls) if p]
    if limit:
        products = [p for p in products if isinstance(p["price_inr"], (int, float)) and p["price_inr"] <= limit]
        out["price_limit_inr"] = limit
    products.sort(key=lambda p: not p["in_stock"])  # things you can actually buy come first
    out["products"] = products[:MAX_PRODUCTS]
    if product_urls or (wanted and not services_only):
        out["compare"] = compare_links(query)

    page_urls = _best(_sitemap("static"), wanted, 2) if wanted else []
    shopping = bool(wanted) and not services_only and not page_urls
    if shopping and not out["products"]:
        page_urls = _best(_sitemap("categories") + _sitemap("brands"), wanted, 2)
        out["note"] = ("No matching product was found on SniffyNose" +
                       (f" under {limit:.0f} rupees." if limit else ".") + " Say so.")
        page_urls = page_urls or [SITE + "/products"]
    if not out["products"] and not page_urls:  # a general question: what is SniffyNose?
        page_urls = [SITE + "/", SITE + "/why-sniffynose"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        out["pages"] = [p for p in pool.map(_read_page, page_urls[:3]) if p]

    if re.search(r"nest|host|board|stay|day ?care", query, re.I):
        out["sniffynest"] = {"title": "SniffyNest: pet hosting by SniffyNose", "url": NEST,
                             "note": ("SniffyNest's own pages are not read by this assistant. Share the link."
                                      if not get_settings().nest_allowed else "Read it with read_webpage.")}
    if not (out["products"] or out["pages"] or out.get("sniffynest") or out.get("note")):
        raise live.ToolError("Nothing matching was found on SniffyNose.")
    return out
