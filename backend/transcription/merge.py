"""Finishing the merged book (REFACTOR_PLAN.md step 76: was the end of the output section of transcribe.py, which
still has the names).

move_catchwords takes a catchword the model wrote into the running text out again (trailing_catchword: one appended
to the last line). The paragraph starts are the model's own marks, as it read them on the scan: nothing measured in
the PDF's text layer is added to them or takes them away (the indent measured there, mark_paragraphs until 12
November 2026, put a paragraph on verse lines, title pages and in the middle of sentences more often than right).
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.bookfile import letters  # noqa: E402


def trailing_catchword(line: ET.Element, follow: str) -> str:
    """The model sometimes appends the catchword to the last line of text ("... diligenter has").
    When the last one to three words of that line are what the next scan begins with, cut them
    off the line and return them. Only a plain text ending is touched, never a word in markup."""
    if line.get("type") == "heading":
        return ""
    holder, attr = (line[-1], "tail") if len(line) else (line, "text")
    words = (getattr(holder, attr) or "").rstrip().split(" ")
    total = len(" ".join("".join(line.itertext()).split()).split())
    for n in (3, 2, 1):
        if len(words) < n or total <= n:
            continue
        cand = words[-n:]
        key = letters("".join(cand))
        if len(key) >= 2 and follow.startswith(key) and not cand[0].endswith(("-", "=")) \
                and not any(w.endswith((".", ":", ";", "?", "!")) for w in cand[:-1]):
            setattr(holder, attr, " ".join(words[:-n]).rstrip() + (" " if attr == "tail" else ""))
            return " ".join(cand)
    return ""


def move_catchwords(book: ET.Element):
    """The model sometimes puts the catchword (the first word(s) of the next page, printed below
    the last line) in the running text as well. A short last line (at most three words) that the
    next scan begins with is the catchword: it goes to <catchword> and the page gets a warning.
    Two such lines (the catchword written twice) both go. A catchword line's own para mark is an
    artefact of where it is printed, so it is not passed on: the next scan's first line starts a
    paragraph only when that scan shows it."""
    pages = [pg for pg in book.iter("page") if pg.find("main_text/l") is not None]
    for pg, nxt in zip(pages, pages[1:]):
        if int(nxt.get("n")) != int(pg.get("n")) + 1:
            continue
        follow = letters(" ".join("".join(l.itertext()) for l in nxt.findall("main_text/l")[:2]))
        main = pg.find("main_text")
        moved = []
        while len(main) > 1:
            last = main[-1]
            text = " ".join("".join(last.itertext()).split())
            key = letters(text)
            if last.get("type") == "heading" or len(text.split()) > 3 or len(key) < 2 \
                    or not follow.startswith(key):
                break
            moved.insert(0, text)
            main.remove(last)
        if not moved and len(main):
            tail = trailing_catchword(main[-1], follow)
            if tail:
                moved.append(tail)
        if not moved:
            continue
        bm = pg.find("bottom_margin")
        if bm is None:
            bm = ET.SubElement(pg, "bottom_margin")
        if not (bm.findtext("catchword") or "").strip():
            cw = bm.find("catchword")
            if cw is None:
                cw = ET.SubElement(bm, "catchword")
            cw.text = moved[-1]
        n = len(main)
        for note in pg.iterfind("marginalia/note"):
            for k in ("line_from", "line_to"):
                if (note.get(k) or "").isdigit() and int(note.get(k)) > n:
                    note.set(k, str(n))
        w = ET.Element("warning", type="catchword_in_text")
        w.text = f"catchword taken out of the main text: {' / '.join(moved)}"
        pg.insert(0, w)
