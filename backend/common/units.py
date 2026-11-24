"""A book as the paragraphs and headings it is read in: the units that are translated, searched and linked.

These stood in translation/translate.py. The register, the loci and the Opera Calvini took them from there and
loaded the whole translation script (and the model's library) to read a book.
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from common.bookfile import line_markup
from common.notice import notice_pages

TAG = re.compile(r"</?(?:i|hi)>")
# Level-1 headings that close a chapter rather than open one.
END_MARKERS = re.compile(r"^(DIXI|FINIS)\b", re.I)


def plain(markup: str) -> str:
    """Text without the <i>/<hi> italics tags."""
    return TAG.sub("", markup)


def letters(markup: str) -> int:
    """Number of letters in a text, as sitebuild/edition/build_reader.py counts them (no digits,
    punctuation, spaces or tags), so hyphenation and spacing do not shift an offset."""
    return len(re.sub(r"[\W\d_]+", "", plain(markup)))


class Unit:
    """A paragraph (heading 0) or a heading (its level) of the Latin text."""

    def __init__(self, page: int, line: int, heading: int, la: str):
        self.start, self.end, self.heading, self.la = [page, line], [page, line], heading, la
        # every printed line of the unit: [scan page, line n, letters of the Latin before it]
        self.lines = [[page, line, 0]]
        self.block, self.sha = None, None   # blocks.sync()

    @property
    def id(self) -> str:
        return f"{self.start[0]}.{self.start[1]}"

    def words(self) -> int:
        return len(plain(self.la).split())

    def add_line(self, page: int, line: int, text: str):
        """Join the next printed line, rejoining a word hyphenated at the line end."""
        self.lines.append([page, line, letters(self.la)])
        m = re.search(r"[-¬⸗]((?:</(?:i|hi)>)*)$", self.la)
        if m and not self.heading:
            self.la = re.sub(r"</(i|hi)><\1>", "", self.la[:m.start()] + m.group(1) + text)
        else:
            self.la += " " + text
        self.la = re.sub(r"</(i|hi)>(\s+)<\1>", r"\2", self.la)  # one italic stretch over two lines
        self.end = [page, line]

    def json(self, tr: str, req: str) -> dict:
        return {"id": self.id, "block": self.block, "la_sha": self.sha, "start": self.start, "end": self.end,
                "lines": self.lines, "heading": self.heading, "la": self.la, "tr": tr, "req": req}


def balance(units: list[Unit]):
    """Italics running on into the next unit: close the tag at the end, reopen it in the next unit."""
    open_tags: list[str] = []
    for u in units:
        u.la = "".join(f"<{t}>" for t in open_tags) + u.la
        stack = []
        for m in re.finditer(r"<(/?)(i|hi)>", u.la):
            if m.group(1):
                if stack and stack[-1] == m.group(2):
                    stack.pop()
            else:
                stack.append(m.group(2))
        u.la += "".join(f"</{t}>" for t in reversed(stack))
        open_tags = stack
        u.la = re.sub(r"<(i|hi)>\s*</\1>", "", u.la).strip()


def read_units(xml: Path) -> list[Unit]:
    """The units of book.xml in reading order.

    A paragraph starts at a line with para="true" and runs on over page breaks until the
    next paragraph or heading. A heading printed over several lines of one page is one
    unit; a chapter heading repeated at the top of the next page (before any text) is
    left out. Title and contents pages: one unit per printed line. The scanner's notice
    in front of the book (Google Books) is not part of the book."""
    units: list[Unit] = []
    chapter, body_since = None, True  # caput of the open chapter; text since its heading?
    root = ET.parse(xml).getroot()
    cover = notice_pages(root)   # the scanner's cover sheet (e-rara) is not part of the book
    for page in root.iter("page"):
        n, kind = int(page.get("n")), page.get("type") or "text"
        if n in cover:
            continue
        lines = list(page.iterfind("main_text/l"))  # (a page can have several <main_text>)
        if kind == "empty" or "books.google" in "".join(t for l in lines for t in l.itertext()):
            continue
        for i, l in enumerate(lines, 1):
            text = line_markup(l)
            if not plain(text).strip():
                continue
            ln = int(l.get("n")) if (l.get("n") or "").isdigit() else i
            last = units[-1] if units else None
            if l.get("type") == "heading":
                level = int(l.get("level") or 2)
                same_heading = last and last.heading == level and last.end[0] == n
                if level == 1 and not END_MARKERS.match(plain(text)):
                    if l.get("caput") and l.get("caput") == chapter and not body_since and not same_heading:
                        continue  # repeated chapter heading across a page break
                    chapter, body_since = l.get("caput"), False
                if same_heading:
                    last.add_line(n, ln, text)  # heading printed over several lines
                else:
                    units.append(Unit(n, ln, level, text))
            else:
                body_since = True
                if kind != "text" or l.get("para") == "true" or last is None or last.heading:
                    units.append(Unit(n, ln, 0, text))
                else:
                    last.add_line(n, ln, text)
    balance(units)
    return units
