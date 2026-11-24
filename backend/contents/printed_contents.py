"""The book's own table of contents, read from its transcribed contents pages, as a contents list.

Many books print one: INDEX CAPITUM, ELENCHUS LOCORUM, LOCORUM TABULA. The transcription keeps
those pages line by line (page type="contents"), with the page numbers as printed. register()
reads the entries off them and finds the scan of each page number from the printed page numbers
of the scans. That is the author's (or printer's) own division of the book, with the full titles,
which no running head gives.

Rules, from how the registers are printed:
- A register is a run of contents pages (with at most GAP other scans between them, often one the
  transcription gave another type).
- An entry is a title with a page number at the end ("De Deo. 12", "pag. 12", "ibid." for the one
  before). A title broken over two lines (hyphen, or no full stop) is joined; a line without a
  number before the entries it heads ("CAPITA LIBRI PRIMI.", "De miseria hominis.") is a group
  above them, at the page of its first entry.
- Two-column registers (cells joined by " | " in the transcription) are read column by column.
  Grids of three columns or more are not used: their cells come out of order and broken.
- A subject index is no table of contents: its numbers jump back and forth (alphabetical order),
  or carry a second number (page and section, "278. 9."), and it stands behind the text it lists.
  A register counts when it has MIN_ENTRIES entries at least, its numbers mostly go up (MONOTONE),
  most of them are found among the scans (FOUND), and it comes before most of the pages it lists.
"""

import re

from common.text import element_text as plain

GAP = 2             # scans of another type allowed inside a run of contents pages
MIN_ENTRIES = 5     # fewer entries with a page number: no register
MONOTONE = 0.7      # share of the entries whose number is not below the one before
FOUND = 0.7         # share of the entries whose page is found among the scans
INDEX_SHARE = 0.3   # share of titles ending in a number (page and section) that makes it an index
TITLE_MAX = 110     # longer titles are cut at a word
GRID = 0.1          # share of lines with three cells or more that makes a register a grid, not read

REF = re.compile(r"^(.*?)[\s.,;:…·]*(?:\b(?:pag|p|fol)\.?\s*)?(\d{1,4}|ibid(?:em)?)\.?\s*$", re.I)
# a register's own title, not a group of entries
OWN_TITLE = re.compile(r"^(?:[IVX]\s+)?(?:i\s*n\s*d\s*e\s*x|index|tabula|elenchus|catalogus|synopsis|ordo|series|"
                       r"syllabus|conspectus|summaria|capitum|locorum|sequuntur|continentur)\b", re.I)
THEIR_TITLE = re.compile(r"\b(?:index|indices|tabula|elenchus|catalogus|ordo|contentorum|continentur|"
                         r"comprehenduntur|habentur|tractantur)\b", re.I)


def printed_int(s: str):
    """The (first) printed page number as an int, or None (roman, a signature, unread):
    "13[?]" is 13, a scan of two pages "3 / 4" is 3."""
    m = re.match(r"\s*(\d{1,4})\b", s or "")
    return int(m.group(1)) if m else None


def smoothed(scans: list[tuple[int, int | None]]) -> list[tuple[int, int | None, int]]:
    """[(scan index, page number, pages per scan)] with the page numbers made consistent: per scan
    the median of what its neighbours (WINDOW scans either way) say it should be, so a misread
    number (10 for 20) or an unread one is put right; pages per scan: 2 for scans of a spread."""
    known = [(i, p) for i, p in scans if p is not None]
    steps = sorted((q - p) / (j - i) for (i, p), (j, q) in zip(known, known[1:]) if 0 < j - i <= 3 and q > p)
    rate = 2 if steps and steps[len(steps) // 2] >= 1.5 else 1
    out = []
    for k, (i, _) in enumerate(scans):
        says = sorted(p - rate * (j - i) for j, p in scans[max(0, k - WINDOW):k + WINDOW + 1] if p is not None)
        out.append((i, says[len(says) // 2] if len(says) >= 2 else None, rate))   # one stray number says little
    return out


WINDOW = 4
NUMBERING_SLIP = 2   # pages a numbering may jump between two scans and still be the same numbering


def cells(lines: list[str]) -> list[str]:
    """The cells of a page in reading order: a two-column register comes out column by column."""
    rows = [[c.strip() for c in l.split(" | ")] for l in lines]
    width = max((len(r) for r in rows), default=1)
    if width == 1:
        return [r[0] for r in rows]
    return [r[c] for c in range(width) for r in rows if len(r) > c]


def joins(prev: str, cell: str) -> bool:
    """Does cell go on the title in prev (broken over two lines)?"""
    return prev.endswith(("-", "=", "¬")) or not prev.rstrip().endswith((".", ":", ";", "?", "!")) \
        or cell[:1].islower() or cell[:1] in "&("


def join(prev: str, cell: str) -> str:
    if prev.endswith(("-", "=", "¬")):
        return prev[:-1] + cell
    return f"{prev} {cell}" if prev else cell


def cut(title: str) -> str:
    title = title.strip(" .,;:-|")
    if len(title) <= TITLE_MAX:
        return title
    return title[:TITLE_MAX].rsplit(" ", 1)[0].rstrip(" ,;:") + " …"


def entries(lines: list[str]) -> list[list]:
    """[[level, title, page number or None]] of a run's lines: groups at level 1 (their number
    is that of their first entry), entries at level 2; without groups all at level 1."""
    out, buf, last = [], "", None

    def group():
        nonlocal buf
        # before the first entry: the register's title too ("LIBRORVM … HOC VOLVMINE CONTENTORVM ORDO")
        own = OWN_TITLE.match(buf) or not out and THEIR_TITLE.search(buf)
        if buf and not own and re.search(r"[A-Za-z]{3}", buf):
            out.append([1, cut(buf), None])
        buf = ""

    for cell in cells(lines):
        cell = " ".join(cell.replace("&nbsp;", " ").replace("&nbsp", " ").split())
        if not cell or cell.startswith("[illegible"):
            continue
        m = REF.match(cell)
        if m and not m.group(1).strip() and buf:   # the number on a line of its own, under its title
            m = REF.match(f"{buf} {cell}")
            buf = ""
        if m and m.group(1).strip():
            ref = last if m.group(2).lower().startswith("ibid") else int(m.group(2))
            title = m.group(1)
            if buf and joins(buf, title):
                title = join(buf, title)
                buf = ""
            else:
                group()
            out.append([2, title, ref])
            last = ref
        elif buf and joins(buf, cell):
            buf = join(buf, cell)
        elif not buf and out and (cell[:1].islower() or out[-1][1].endswith(("-", "=", "¬"))):
            out[-1][1] = join(out[-1][1], cell)   # the rest of a title, printed under its number
        else:
            group()
            buf = cell
    group()
    # a group's number is that of its first entry; a group with no entries goes
    for i, e in enumerate(out):
        if e[0] == 1:
            e[2] = next((f[2] for f in out[i + 1:] if f[0] == 2), None)
    out = [e for i, e in enumerate(out)
           if e[0] == 2 or (i + 1 < len(out) and out[i + 1][0] == 2)]
    if not any(e[0] == 1 for e in out):
        for e in out:
            e[0] = 1
    for e in out:
        e[1] = cut(e[1])
    return out


def is_register(es: list[list]) -> bool:
    numbered = [e for e in es if e[2] is not None]
    refs = [e[2] for e in numbered]
    if len(refs) < MIN_ENTRIES:
        return False
    up = sum(b >= a for a, b in zip(refs, refs[1:])) / (len(refs) - 1)
    ending = sum(bool(re.search(r"\d[.,]?$", e[1])) for e in numbered) / len(numbered)
    # the letters of an alphabetical index: "A." on a line of its own, or before the entry ("S Sabbathum")
    # (not a roman number: "V. De Summo Bono")
    letters = sum(bool(re.fullmatch(r"[A-Z]\.?", e[1]) or re.match(r"(?![IVXLCDM])[A-Z]\.? [A-Z][a-z]", e[1]))
                  for e in es)
    return up >= MONOTONE and ending < INDEX_SHARE and letters < 3


def locate(es: list[list], scans: list[tuple[int, int | None]], parts_after: int | None = None) -> list[tuple[int, int, str]]:
    """[(level, scan index, title)] for the entries whose page is found. Read from the last entry
    back: each is the last scan that holds its page (by the smoothed numbers) at or before the one
    found for the entry after it. So a preface numbered 1, 2, … of its own is passed over for the
    book's page 1, and a second part numbered from 1 again is found in that part.

    An entry with a higher number than the one after it is one of two things. An earlier part numbered
    on its own ends here (part one before part two) when all of this holds: it is the entry right
    before the one that was found (the list passes from one part to the next between two entries; once
    one entry was a stray, those before it are strays too, until one is in line again); its page stands
    before the place we are at, in a numbering that begins after scan parts_after, the last page of
    the list itself (a list names what comes after it: front matter numbered on its own, with the list
    in it, is no part); the entry before it does
    not simply go on below the one after it (it lies beyond that one too, or it is the first of its
    own part); and that entry's page is found in the same earlier part. The entry is then found there,
    and the entries before it follow it, after the list as well. Else it is a stray number (cells of a grid read out of order,
    a misprint): looked up anywhere, and it does not hold the entries before it back. Without
    parts_after (an index, which is in no order of pages) every such entry is a stray."""
    pages, out, at, after = smoothed(scans)[::-1], [], float("inf"), None
    known = [e for e in es if e[2] is not None]
    in_line = True   # the entry after this one was found in line: this one may end an earlier part
    low = float("-inf")   # once a part is found: the entries in line before it lie after the list too

    def last(ref: int, low: float, high: float) -> int | None:
        """The last scan after low and up to high that holds page ref."""
        return next((i for i, p, rate in pages if low < i <= high and p is not None and p <= ref < p + rate), None)

    def begins(hit: int) -> float:
        """The scan on which the numbering that scan `hit` stands in has its page 1 (or begins, if later)."""
        first = None
        for i, p, rate in pages:   # from the last scan back
            if i > hit or p is None:
                continue
            if first is not None and abs((first[0] - i) * rate - (first[1] - p)) > NUMBERING_SLIP:
                break
            first = (i, p)
        return first[0] + max(0, 1 - first[1]) / pages[0][2]

    for k in range(len(known) - 1, -1, -1):
        level, title, ref = known[k]
        inverted = after is not None and ref > after
        hit = earlier = None
        if inverted and in_line and parts_after is not None and k:
            before = known[k - 1][2]
            if before <= ref and (before > after or k == 1 or known[k - 2][2] > before):
                earlier = last(ref, parts_after, at)
                if earlier is not None and (last(before, parts_after, earlier) is None or begins(earlier) <= parts_after):
                    earlier = None
                hit = earlier
                low = parts_after if earlier is not None else low
        if hit is None and not inverted:
            hit = last(ref, low, at)
            inverted = hit is None and low == parts_after   # not in the part that was found: a stray after all
        if hit is None and inverted:
            hit = last(ref, float("-inf"), float("inf"))
        if hit is not None:
            out.append((level, hit, title))
            if not inverted or earlier is not None:
                at, after = hit, ref
        if inverted and earlier is None:
            in_line = False   # a stray
        elif hit is not None:
            in_line = True
    return sorted(out[::-1], key=lambda e: e[1])   # register order within a scan


def register(root) -> list[tuple[int, int, str]]:
    """The book's own contents list: [(level, scan index, title)], or [] if it has none that holds."""
    pages = list(root.iter("page"))
    scans = [(int(p.get("n")), printed_int(p.findtext("page_number"))) for p in pages]
    runs, run = [], []
    for p in pages:
        n = int(p.get("n"))
        if p.get("type") == "contents":
            if run and n - run[-1] > GAP + 1:
                runs.append(run)
                run = []
            run.append(n)
    if run:
        runs.append(run)
    by_n = {int(p.get("n")): p for p in pages}
    best = []
    for run in runs:
        lines = [plain(l) for n in range(run[0], run[-1] + 1) if n in by_n
                 for l in by_n[n].iterfind("main_text/l")]
        # a grid of three columns or more (W07109's tables of loci) is read out of order
        if sum(l.count(" | ") >= 2 for l in lines) >= GRID * len(lines):
            continue
        es = entries(lines)
        if not is_register(es):
            continue
        found = locate(es, scans, run[-1])
        # a table of contents comes before the text it lists, a subject index after it
        if not found or run[0] > sorted(f[1] for f in found)[len(found) // 2]:
            continue
        # pages beyond the last scan (a book transcribed in part) do not count against it
        last = max((p for _, p, _ in smoothed(scans) if p is not None), default=0)
        if len(found) >= FOUND * sum(e[2] is not None and e[2] <= last for e in es) and len(found) > len(best):
            best = found
    return best
