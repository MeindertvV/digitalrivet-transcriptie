"""Where a sentence ends: the one splitter of sentences_<lang>.json (sentence_units.py) and the interlinear text
(interlinear.py), REFACTOR_PLAN.md step 91. Behind sentences() alone, so another splitter can take its place (P42).

A sentence ends at . ? or ! before a capital, not after a short word or a number ("Cor." "Tho." "3.") and not
before a digit ("Psal. 52. 8." is a reference); a sentence longer than LONG characters is cut at : and ;.
"""

import re

END = re.compile(r"(?<=[.?!])(?:</?i>|[\"”')\]])*\s+(?=(?:<i>|[\"“'(\[])*[A-ZÆŒ])")   # not before a digit: "Psal. 52. 8." is a reference
SHORT = re.compile(r"(?:^|\s)(?:\w{1,3}|[A-Z]\w?|\d+)\.$")   # "Cor." "Tho." "3." do not end a sentence


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of the sentences of text."""
    spans, start = [], 0
    for m in END.finditer(text):
        head = text[start:m.start()]   # ends with the . ? or !
        if SHORT.search(head):
            continue
        spans.append((start, m.start() + len(m.group(0).rstrip())))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return [c for a, b in spans if text[a:b].strip() for c in clauses(text, a, b)]


LONG, CLAUSE = 250, 35   # a sentence longer than LONG is cut at : and ; into parts of at least CLAUSE characters
COLON = re.compile(r"[:;](?:</i>)?\s+")


def clauses(text: str, a: int, b: int) -> list[tuple[int, int]]:
    """A sentence without full stops (a long run of clauses) cut at its colons and semicolons."""
    if b - a <= LONG:
        return [(a, b)]
    out, start = [], a
    for m in COLON.finditer(text, a, b):
        if m.end() - start >= CLAUSE and b - m.end() >= CLAUSE:
            out.append((start, m.start() + len(m.group(0).rstrip())))
            start = m.end()
    out.append((start, b))
    return out


def texts(text: str) -> list[str]:
    """The sentences of text themselves."""
    return [text[a:b] for a, b in sentences(text)]
