"""Cut a streaming answer into speakable pieces, so speech can start before the whole answer is written."""
from __future__ import annotations

import re

FIRST_MIN = 24  # the opening piece may end at a comma once it is this long, to start talking sooner
MIN_PIECE = 12  # very short sentences are joined to the next one
MAX_PIECE = 220  # a very long sentence is cut at a comma or space

_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "rs", "mt", "shri", "smt"}
_END = re.compile(r"[.!?।…]+[\"'”’)\]]*(?=\s)")


def _is_real_end(text: str, match: re.Match) -> bool:
    if not match.group(0).startswith("."):
        return True
    word = re.search(r"([A-Za-z]+)$", text[: match.start()])
    # "Dr. Rao", "N. Chandrababu" and "e.g. this" are not sentence ends
    return not (word and (len(word.group(1)) == 1 or word.group(1).lower() in _ABBREVIATIONS))


class SentenceStream:
    def __init__(self):
        self.buffer = ""
        self.emitted = 0

    def _cut(self) -> int | None:
        text = self.buffer
        def inside_mark(position: int) -> bool:  # never cut inside a 【citation】
            return text.count("【", 0, position) > text.count("】", 0, position)

        for m in _END.finditer(text):
            if m.end() >= MIN_PIECE and _is_real_end(text, m) and not inside_mark(m.end()):
                return m.end()
        if self.emitted == 0:
            comma = re.search(r"[,;:](?=\s)", text[FIRST_MIN:])
            if comma and not inside_mark(FIRST_MIN + comma.end()):
                return FIRST_MIN + comma.end()
        if len(text) > MAX_PIECE and not inside_mark(MAX_PIECE):
            head = text[:MAX_PIECE]
            cut = max(head.rfind(", "), head.rfind("; "))
            cut = cut + 1 if cut > MIN_PIECE else head.rfind(" ")
            return cut if cut > 0 else MAX_PIECE
        return None

    def feed(self, piece: str) -> list[str]:
        """Add newly received text; return any pieces that are now complete."""
        self.buffer += piece or ""
        done = []
        while (cut := self._cut()) is not None:
            part, self.buffer = self.buffer[:cut].strip(), self.buffer[cut:].lstrip()
            if part:
                done.append(part)
                self.emitted += 1
        return done

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer.strip(), ""
        return [rest] if rest else []
