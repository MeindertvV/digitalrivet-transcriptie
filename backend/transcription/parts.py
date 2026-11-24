"""A hard page in parts, each read in a request of its own, and the answers joined into one page.

Haiku does not misread a page with columns or thick margins much more than another page, but it
leaves things out: margin notes, the last column, the block below the text. Given half a page it
keeps them (on hard pages five times the margin notes), at twice the price of such a page. So
plan() looks at the OCR word positions, no API needed, which also makes it work in a batch:

- a scan wider than high is two printed pages: left and right;
- columns (pdfboxes.gutters): one part per column, cut at the gap between them. What runs over the
  full width above the columns goes with the first part and what runs below them with the last,
  so the reading order stays right: the other columns are blanked out in those parts;
- one column with heavy margin notes, a table or a list (layout_check.page_side): cut across at
  the widest gap between two lines near the middle of the page.

Any other page, and every page without a text layer, stays whole. join() makes one <page> of the
answers, so what comes after it does not know there were parts.

check() is the other half: it looks at the finished page and warns when much less text came out
than the OCR layer holds, when the answer is doubtful nearly everywhere, or when the model gave up
on a block. Those are the pages that look fine and are not.
"""

import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import lineboxes  # noqa: E402  (gutters, norm)
from transcription import layout_check  # noqa: E402

MIN_WORDS = 80        # fewer OCR words: a title page, a plate; nothing to gain
SPREAD = 1.15         # a scan this much wider than high holds two printed pages
HEAVY = 0.08          # share of the letters beside the main column that makes a page hard
MAX_PARTS = 4         # more columns than this: the gaps are a table's, not columns
SAME_GAP = 0.1        # share of the page's width: gaps this close are one gap
COLUMN = 0.15         # share of the letters a column must hold on either side of a gap
JOIN_GAP = 0.02       # share of the page's width: a gap this small at a gutter is no gap between columns
MIDDLE = (0.35, 0.65) # where a cut across the page may fall
RUN = 3               # lines in a row that keep to their column: the columns have started
FOOT = 0.05           # share of the printed height: columns that end this close to its foot run to the foot

SHORT = 0.7           # share of the OCR layer's letters a page must have, or it gets a warning
MIN_LETTERS = 400     # fewer letters in the OCR layer: too little to compare
DOUBTFUL = 0.3        # share of the words with a [?] that makes a page doubtful
MARKER = re.compile(r"\[illegible[^\]]*\]|\[\?\]")
LOST_LINES = 3        # lines that hold nothing but a marker: a block was given up, not a word
REMARK = re.compile(r"\.\.\.|…|not transcribed|remaining lines|see below|cut off", re.I)  # banned in the text


@dataclass
class Part:
    """A part of a page in PDF points: what to render, and what in it to blank out."""
    clip: tuple[float, float, float, float]
    blank: list[tuple[float, float, float, float]] = field(default_factory=list)


from common.lineboxes import word_rows as ocr_rows  # noqa: E402,F401  (step 65)


def crosses(ws, gx: list[float], width: float) -> bool:
    """Does this line run over a gap between columns? A word lies on the gap, or the words on either
    side of it stand too close to be two columns (the OCR can put two columns on one line)."""
    for g in gx:
        if any(w[0] < g < w[2] for w in ws):
            return True
        left = [w[2] for w in ws if w[2] <= g]
        right = [w[0] for w in ws if w[0] >= g]
        if left and right and min(right) - max(left) < JOIN_GAP * width:
            return True
    return False


def columns(rows, width: float) -> list[float]:
    """The gaps between real columns. pdfboxes.gutters also finds the edge of the text beside a thin
    margin, a strip of the neighbouring page or a column of numbers: a real gap has much text on
    both sides. Gaps close together are one."""
    gx = []
    for g in lineboxes.gutters(rows, width):
        if gx and g - gx[-1][-1] < SAME_GAP * width:
            gx[-1].append(g)
        else:
            gx.append([g])
    gx = [sum(run) / len(run) for run in gx]
    words = [((w[0] + w[2]) / 2, len(w[4])) for ws in rows for w in ws]
    total = sum(n for _, n in words) or 1
    return [g for g in gx if COLUMN * total <= sum(n for x, n in words if x < g) <= (1 - COLUMN) * total]


def bands(rows, gx: list[float], width: float, height: float) -> tuple[float, float]:
    """Where the columns start and end on the page: above and below that, text runs the full width."""
    rows = sorted(rows, key=lambda ws: ws[0][1])
    apart = [not crosses(ws, gx, width) and len(ws) > 2 for ws in rows]
    top, bottom = 0.0, height
    for i in range(len(rows) - RUN + 1):
        if all(apart[i:i + RUN]):
            top = max(min(w[1] for w in rows[i]) - 2, 0)
            break
    for i in range(len(rows) - 1, RUN - 2, -1):
        if all(apart[i - RUN + 1:i + 1]):
            bottom = min(max(w[3] for w in rows[i]) + 2, height)
            break
    return top, bottom


def under_heading(rows, gx: list[float], top: float, bottom: float, width: float) -> bool:
    """Columns that fill half the print, run on to its foot, and stand under text that runs over the
    gap: most letters above them lie in lines that cross it. Columns from the top of the page where
    bands() starts late (an index, a page of verse) have their own lines up there and stay whole."""
    y0 = min(w[1] for ws in rows for w in ws)
    y1 = max(w[3] for ws in rows for w in ws)
    if y1 - bottom >= FOOT * (y1 - y0) or bottom - top < 0.5 * (y1 - y0):
        return False
    above = [ws for ws in rows if max(w[3] for w in ws) <= top]
    n = sum(len(w[4]) for ws in above for w in ws)
    over = sum(len(w[4]) for ws in above if crosses(ws, gx, width) for w in ws)
    return n > 0 and over >= 0.5 * n


def column_parts(gx: list[float], top: float, bottom: float, width: float, height: float) -> list[Part]:
    xs = [0.0] + gx + [width]
    last = len(xs) - 2
    out = []
    for k, (a, b) in enumerate(zip(xs, xs[1:])):
        y0 = 0.0 if k == 0 else top
        y1 = height if k == last else bottom
        x0 = 0.0 if k == last and bottom < height else a
        x1 = width if k == 0 and top > 0 else b
        blank = []
        if k == 0 and top > 0 and x1 > b:
            blank.append((b, top, width, y1))
        if k == last and bottom < height and a > 0:
            blank.append((0.0, y0, a, bottom))
        out.append(Part((x0, y0, x1, y1), blank))
    return out


def gap_across(rows, height: float) -> float:
    """The middle of the widest gap between two lines near the middle of the page."""
    boxes = sorted((min(w[1] for w in ws), max(w[3] for w in ws)) for ws in rows)
    best, y = -1.0, height / 2
    reach = boxes[0][1] if boxes else 0
    for y0, y1 in boxes[1:]:
        if MIDDLE[0] * height < y0 < MIDDLE[1] * height and y0 - reach > best:
            best, y = y0 - reach, (y0 + reach) / 2
        reach = max(reach, y1)
    return y


def plan(page) -> list[Part]:
    """How to read a pymupdf page with a text layer: its parts, or [] to keep it whole."""
    width, height = page.rect.width, page.rect.height
    if len(page.get_text("words")) < MIN_WORDS:
        return []
    rows = ocr_rows(page)
    gx = columns(rows, width)
    if width > SPREAD * height:
        mid = min(gx, key=lambda g: abs(g - width / 2)) if gx else width / 2
        return [Part((0.0, 0.0, mid, height)), Part((mid, 0.0, width, height))]
    if gx and len(gx) < MAX_PARTS:
        top, bottom = bands(rows, gx, width, height)
        # Under half the page: a table the OCR reads across, so columns over the whole page. Unless
        # under_heading: then the scan's white margin made the columns look short, and the long
        # heading above them was cut in two (W00943 p. 4).
        if bottom - top < 0.5 * height and not under_heading(rows, gx, top, bottom, width):
            top, bottom = 0.0, height
        return column_parts(gx, top, bottom, width, height)
    side = layout_check.page_side(page)
    if side is not None and HEAVY <= side < 1:   # 1: columns the OCR ran together; a cut across would mix them
        y = gap_across(rows, height)
        return [Part((0.0, 0.0, width, y)), Part((0.0, y, width, height))]
    return []


def request_text(k: int, n: int) -> str:
    """The request for part k (1-based) of n."""
    text = (f"This image is part {k} of {n} of one page: the rest of the page is cut off or blanked out. "
            "Transcribe only this part, as if it were the page, and say nothing about the cut in <notes>.")
    if k < n:
        text += " The page goes on after it: no catchword or signature in this part."
    if k > 1:
        text += (" It starts in mid-page: a page number or running head only if you see one, and the first "
                 'paragraph gets cont="true" unless it clearly starts a new one.')
    return text


# ---------------------------------------------------------------- joining

def add_text(el: ET.Element, text: str | None):
    """Append text at the end of el."""
    if not text:
        return
    if len(el):
        el[-1].tail = (el[-1].tail or "") + text
    else:
        el.text = (el.text or "") + text


def end_of(el: ET.Element) -> str:
    return (el[-1].tail if len(el) else el.text) or ""


def continue_block(block: ET.Element, more: ET.Element):
    """more goes on where block stops; a word broken over the cut stays one word."""
    end = end_of(block).rstrip()
    broken = end.endswith(("-", "=", "¬"))
    if len(block):
        block[-1].tail = end if broken else end + " "
    else:
        block.text = end if broken else end + " "
    add_text(block, more.text)
    for child in list(more):
        block.append(child)


def unique(values: list[str]) -> str:
    """The texts of the parts, each once: the same running head twice is one, two numbers are "3 / 4"."""
    seen = []
    for v in values:
        if v and v not in seen:
            seen.append(v)
    return " / ".join(seen)


def join(pages: list[ET.Element]) -> str:
    """One <page> from the answers of the parts, in reading order. The page type is the first that
    says more than "empty"; catchword and signature come from the last part that has them."""
    full = [p for p in pages if p.get("type") != "empty"] or pages[:1]
    page = ET.Element("page", {k: v for k, v in full[0].attrib.items()})
    for w in (w for p in pages for w in p.findall("warning")):
        page.append(w)

    def texts(path):
        return [" ".join("".join(e.itertext()).split()) for p in full for e in p.findall(path)]

    for tag in ("layout", "page_number", "running_head"):
        value = "; ".join(texts(tag)) if tag == "layout" else unique(texts(tag))
        if value:
            ET.SubElement(page, tag).text = value
    foot = {t: [v for v in texts(f"bottom_margin/{t}") if v] for t in ("catchword", "signature")}
    if foot["catchword"] or foot["signature"]:
        margin = ET.SubElement(page, "bottom_margin")
        for t, values in foot.items():
            if values:
                ET.SubElement(margin, t).text = values[-1]
    main = ET.SubElement(page, "main_text")
    for p in full:
        blocks = list(p.find("main_text")) if p.find("main_text") is not None else []
        if blocks and len(main) and blocks[0].tag == "p" and blocks[0].get("cont") == "true" \
                and main[-1].tag == "p" and not main[-1].get("type"):
            continue_block(main[-1], blocks.pop(0))
        main.extend(blocks)
    notes = " ".join(t for t in texts("notes") if t)
    if notes:
        ET.SubElement(page, "notes").text = notes
    return ET.tostring(page, encoding="unicode")


# ---------------------------------------------------------------- the finished page

from common.text import pairing_letter_count as letters  # noqa: E402,F401  (step 57)


def check(xml: str, ocr_text: str | None) -> tuple[str, str | None]:
    """Warnings on a finished page: <warning type="incomplete"> when much less text came out than the
    OCR layer holds or a block was given up, <warning type="doubtful"> when [?] stands on nearly every
    word or a remark stands in the text. Returns (xml, warning text or None)."""
    page = ET.fromstring(xml)
    if page.get("type") in ("empty", "missing", "blank", "title"):
        return xml, None
    found = []
    written = " ".join("".join(e.itertext()) for tag in ("main_text", "marginalia") for e in page.findall(tag))
    words = written.split()
    plain = MARKER.sub(" ", written)
    have, ocr = letters(plain), letters(ocr_text or "")
    if ocr >= MIN_LETTERS and have < SHORT * ocr:
        found.append(("incomplete", f"only {have / ocr:.0%} as many letters as the OCR layer: a block may be missing"))
    if len(words) >= 20 and written.count("[?]") > DOUBTFUL * len(words):
        found.append(("doubtful", f"{written.count('[?]')} of {len(words)} words marked [?]: read this page with care"))
    blocks = ["".join(e.itertext()).strip() for main in page.findall("main_text") for e in main]
    lost = sum(1 for b in blocks if b and not MARKER.sub("", b).strip())
    if lost >= LOST_LINES:
        found.append(("incomplete", f"{lost} lines hold only a marker: a block was given up"))
    remark = REMARK.search(written)
    if remark:
        found.append(("doubtful", f'"{remark.group(0)}" in the text: a passage was skipped or a remark written as print'))
    if not found:
        return xml, None
    for i, (kind, text) in enumerate(found):
        w = ET.Element("warning", type=kind)
        w.text, w.tail = text, "\n  "
        page.insert(i, w)
    return ET.tostring(page, encoding="unicode"), "; ".join(t for _, t in found)
