"""Show the links and pictures behind an answer. All text here comes from web pages, so it is escaped."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

_MARKDOWN = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")


def plain(text: str) -> str:
    """Web text shown literally: it cannot create links, images or formatting."""
    return _MARKDOWN.sub(r"\\\1", " ".join((text or "").split()))


def link_url(url: str) -> str:
    return url.replace("(", "%28").replace(")", "%29").replace(" ", "%20").replace("<", "%3C").replace(">", "%3E")


def show(cards: list | None, images: list | None) -> None:
    import streamlit as st

    if images:
        columns = st.columns(len(images[:4]))
        for column, url in zip(columns, images[:4]):
            column.image(url, width="stretch")
    if not cards:
        return
    columns = st.columns(2)
    for i, card in enumerate(cards[:4]):
        with columns[i % 2].container(border=True):
            if card.get("image"):
                st.image(card["image"], width="stretch")
            st.markdown(f"**[{plain(card['title'])}]({link_url(card['url'])})**")
            if card.get("snippet"):
                st.caption(plain(card["snippet"]))
            st.caption(f"🔗 {plain((urlsplit(card['url']).hostname or '').removeprefix('www.'))}")
