"""Show the links and pictures behind an answer. All text here comes from web pages, so it is escaped."""
from __future__ import annotations

import html
import re
from urllib.parse import urlsplit

_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")


def plain(text: str) -> str:
    """Web text shown literally: it cannot create links, images or formatting."""
    return _MARKDOWN.sub(r"\\\1", " ".join((text or "").split()))


def link_url(url: str) -> str:
    return url.replace("(", "%28").replace(")", "%29").replace(" ", "%20").replace("<", "%3C").replace(">", "%3E")


def card_html(card: dict) -> str:
    """One card as a single link: picture, title and price all open the page in a new tab."""
    url = card.get("url", "")
    if not url.startswith(("https://", "http://")):
        return ""

    def safe(text: str) -> str:  # escaped for HTML; "$" as an entity so it is never read as maths
        return html.escape(" ".join((text or "").split()), quote=True).replace("$", "&#36;")

    host = (urlsplit(url).hostname or "").removeprefix("www.")
    picture = (f"<img src='{safe(card['image'])}' alt='{safe(card.get('title'))}' loading='lazy'>"
               if str(card.get("image", "")).startswith("https://") else "<span class='pv-noimg'>🔗</span>")
    go = "View on SniffyNose" if host == "sniffynose.com" else f"Open {host}"
    return (f"<a class='pv-card' href='{safe(link_url(url))}' target='_blank' rel='noopener noreferrer'>{picture}"
            f"<span class='pv-title'>{safe(card.get('title'))}</span>"
            f"<span class='pv-sub'>{safe(card.get('snippet'))}</span><span class='pv-go'>{safe(go)} →</span></a>")


def show(cards: list | None, images: list | None, shops: list | None = None) -> None:
    import streamlit as st

    if images:
        columns = st.columns(len(images[:4]))
        for column, url in zip(columns, images[:4]):
            column.image(url, width="stretch")
    tiles = "".join(card_html(c) for c in (cards or [])[:6])
    if tiles:
        st.markdown(f"<div class='pv-cards'>{tiles}</div>", unsafe_allow_html=True)
    if shops:
        st.caption("Compare elsewhere (opens a search on that site):")
        for column, shop in zip(st.columns(len(shops)), shops):
            column.link_button(f"Search {shop['name']}", shop["url"], width="stretch")
