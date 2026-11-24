"""The part of toc_haiku.py that the contents use: reading the lines that may be headings of a book (candidates(),
sheet()), checking an outline against them (check(), like(), outline()), setting it aside when it went wrong in a way
a reader would not notice (doubt()), and reading and applying the toc_haiku.json that was made (toc(), apply()).

REFACTOR_PLAN.md step 36: structure.py, toc_context.py and commentary.py use these; the rest of toc_haiku.py is its
own run (ask Haiku, tidy, fill the gaps), which takes them from here. Moved line for line; the names in toc_haiku.py
are the same objects. Their docstrings still say "here" for toc_haiku.py where they speak of the run.
"""

import collections
import difflib
import json
import re
import sys
import threading
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contents import printed_contents as R  # noqa: E402
from common.paths import DATA, PDF_DIR  # noqa: E402
from common.bookfile import lines_of  # noqa: E402  (step 63)
from common.works import folder_of  # noqa: E402,F401  (step 53: toc_context.py and commentary.py take it from here)

OCR_TOO = False          # transcriptions only, for now: a book that is only OCR keeps the contents it had
OUT_NAME = "toc_haiku.json"
LINE_MAX = 90            # a longer line is no heading unless it was marked as one
CUT = 120                # characters of a line shown on the sheet
HEAD_MEMORY = 3          # a running head like one of the last so many pages is no change
HEAD_PAGES = 40          # scans at the front where a printed contents is looked for
TAIL = 30                # ... and at the back (at least; or a tenth of the book)
TOP = 8                  # lines at the top of a page where its kind shows
CONTENTS_LINES = 350     # lines of printed contents given with the sheet
READ_PX = 1100           # long edge of a heading image
CHAPTER_WORDS = 9        # words in a title at most on average, for a numbered series to be chapters
LEVELS = 5               # levels of the contents at most
DOUBT_THICK = 100        # scans from which a book's contents are compared with what it had
DOUBT_MANY = 20          # entries of the old contents from which that comparison says something
DOUBT_FEWER = 0.4        # share of those the outline must have
DOUBT_SHARE = 0.7        # share of entries that are numbered statements from which the outline is doubted
AFTER = 2                # lines below a bare label shown on the sheet
REPEAT = 4               # a text at the top of this many pages is a running head
MIN_LIKE = 0.45          # share of letters of a title that must agree with its lines


def as_printed(words: str) -> str:
    """A pattern of words that also takes them as the old presses set them: V for u and u for v (CAPVT, QVAESTIO,
    Controuersia), Æ and Œ for ae and oe (PRÆFATIO)."""
    return re.sub(r"ae|oe|[uv]", lambda m: {"ae": "(?:ae|æ)", "oe": "(?:oe|œ)"}.get(m.group(0), "[uv]"), words)


DIVISION = re.compile(r"^\W*(?:" + as_printed(
    "liber|lib|caput|cap|capitulum|pars|part|sectio|sect|quaestio|quaest|qu|disputatio|disp|dissertatio|"
    "exercitatio|exerc|articulus|art|locus|tractatus|tract|concio|homilia|sermo|oratio|epistola|epist|psalmus|"
    "praefatio|praefatiuncula|prooemium|prolegomena|dedicatio|ad lectorem|lectori|index|appendix|"
    "thesis|theses|controversia|contr|tomus|volumen|chapter|book|sermon|section|the first|the second") + r")\b", re.I)
WORD = re.compile(r"[^\W\d_]+")
BOOK_ANY = re.compile(r"\b(?:liber|libri|lib\.|tomus|pars)(?!\w)", re.I)
BACK = re.compile(r"\W*(?:index|errata|finis|appendix|catalogus)", re.I)
LABEL_AT_END = re.compile(r".{3,}?[.,;:]\s+(?:CAP(?:UT|VT)?|LIB(?:ER)?)\.?\s*(?:[IVXLC]+|\d+)\.?$")   # a title with its label after it, in capitals as printed: Y, v, y. CAP. XXXII.
LABEL_LAST = re.compile(r"\W*(.+?)[.,;:]\s+(cap(?:ut|vt|\.)?|lib(?:er|\.)?)\s*([IVXLC]+|\d+|[^\W\d_]+)\W*$", re.I)
BY_CHAPTER = re.compile(r"\W*([^\W\d_]{3,})\b[^.]{0,20}?\bcap(?:ut|\.|itis|ite)?\s*([IVXLC]+|\d+)\b", re.I)
NUMBER_FIRST = re.compile(r"\W*(?:[IVXLCDM]+|\d+)\W+\w")
CONTENTS = re.compile(r"\b(?:i\s?n\s?d\s?e\s?x|elenchus|syllabus|series|tabula|catalogus|conspectus|summa|synopsis|argumenta|"
                      r"capita|contenta|contents|table)\b", re.I)
ALPHABETICAL = re.compile(r"\b(?:rerum|verborum|nominum|locorum|scripturae|alphabet\w*|autorum|auctorum|errata)\b", re.I)


from common.text import heading_fold as folded, ordinal_compound as ordinal, roman_any as roman  # noqa: E402,F401  (step 57)


def running(root) -> set[str]:
    """The folded texts that stand at the top of REPEAT pages or more: running heads in the text. A
    transcription keeps those apart: what it marks as a heading, or a label with a number, is none
    (Caput I. at the top of a page in every book of a work)."""
    ocr = root.get("kind") == "ocr"
    seen = collections.Counter()
    for pg in root.iter("page"):
        for l in lines_of(pg)[:2]:
            text = R.plain(l)
            t = folded(text)
            if 3 <= len(t) <= 60 and (ocr or not (l.get("type") == "heading" or number(text))):
                seen[t] += 1
    return {t for t, n in seen.items() if n >= REPEAT}


def label_only(text: str) -> bool:
    """A division's label with hardly more than its number: the title is on the line below."""
    return bool(DIVISION.match(text)) and len(WORD.findall(text)) <= 3 and len(text) <= 24


def head_changes(root, heads: set[str]) -> dict[int, str]:
    """{scan: running head} where the running head changes: the book's divisions, also where the
    heading itself is not on the sheet. From <running_head>, or a repeated top line of the OCR."""
    out, recent, count = {}, collections.deque(maxlen=HEAD_MEMORY), None
    for pg in root.iter("page"):
        head = " ".join((pg.findtext("running_head") or "").split())
        if not head:
            head = next((" ".join(R.plain(l).split()) for l in lines_of(pg)[:2] if folded(R.plain(l)) in heads), "")
        key = folded(re.sub(r"\d+", "", head))
        if len(key) < 3:
            continue
        num = entry_number({"title": head})
        if (num and num != count) or not any(difflib.SequenceMatcher(None, key, o, autojunk=False).ratio() > 0.75 for o in recent):
            out[int(pg.get("n"))] = head[:CUT]
        count = num or count
        recent.append(key)
    return out


def candidates(root) -> list[dict]:
    """The lines that may be headings: marked ones, short ones that look like a division, the
    AFTER lines below a bare label, which often hold its title but look like text, and where the
    running head changes. [{id, scan, n (line number, 0 the top of the page), text, level (as
    marked, 0 unmarked, -1 below a label, -2 running head)}], without the repeats of running heads."""
    heads = running(root)
    changes = head_changes(root, heads)
    ocr = root.get("kind") == "ocr"
    out = []

    def add(scan, n, text, level, pos=None):
        out.append({"id": len(out) + 1, "scan": scan, "n": n, "text": text[:CUT], "level": level, "pos": pos})

    for pg in root.iter("page"):
        scan = int(pg.get("n"))
        if scan in changes:
            add(scan, 0, changes[scan], -2)
        after = 0
        ls = lines_of(pg)
        for pos, l in enumerate(ls):
            text = " ".join(R.plain(l).split())
            marked = l.get("type") == "heading"
            if not WORD.search(text) or (folded(text) in heads and (ocr or not marked)):
                continue
            letters = [c for c in text if c.isalpha()]
            caps = len(letters) >= 5 and sum(c.isupper() for c in letters) > 0.8 * len(letters)
            if marked or (len(text) <= LINE_MAX and (DIVISION.match(text) or caps or LABEL_AT_END.match(text))):
                add(scan, int(l.get("n") or 0), text, int(l.get("level") or 2) if marked else 0, pos)
                after = AFTER if label_only(text) else max(0, after - 1)
            elif after:
                add(scan, int(l.get("n") or 0), text, -1, pos)
                after -= 1
        for k, h in enumerate(pg.findall("head"), 1):   # a page left in blocks by the model: its headings are marked
            text = " ".join(R.plain(h).split())
            if WORD.search(text):
                add(scan, len(lines_of(pg)) + k, text, int(h.get("level") or 2))
        for c in reversed(out):   # by place, not number: a page whose lines lack numbers has them all 0
            if c["scan"] != scan:
                break
            c["last"] = bool(ls) and c.pop("pos", None) == len(ls) - 1
    return out


def contents_pages(root, most: int = CONTENTS_LINES) -> str:
    """The text of the pages at the front and back that may be a printed table of contents, as a
    checklist for the model: only what they say, cut to most lines."""
    pages = list(root.iter("page"))
    tail = max(TAIL, len(pages) // 10)
    out = []
    for k, pg in enumerate(pages):
        if HEAD_PAGES <= k < len(pages) - tail:
            continue
        ls = [" ".join(R.plain(l).split()) for l in lines_of(pg)]
        top = " ".join([pg.findtext("running_head") or ""] + ls[:TOP])
        if pg.get("list") != "index" and (pg.get("type") == "contents" or (CONTENTS.search(top) and not ALPHABETICAL.search(top))):
            out += [f"[scan {pg.get('n')}]"] + [t[:70] for t in ls if WORD.search(t)]
    return "\n".join(out[:most])


def sheet(root, cands: list[dict]) -> list[str]:
    """The sheet as lines: per scan with candidates its printed page and running head, then the
    candidates: id, line number on the page, H1/H2/H3 as marked, - unmarked, ~ below a label, R a new running head, text."""
    pages = {int(p.get("n")): p for p in root.iter("page")}
    by_scan = collections.defaultdict(list)
    for c in cands:
        by_scan[c["scan"]].append(c)
    out = []
    for scan in sorted(by_scan):
        pg = pages[scan]
        printed = (pg.findtext("page_number") or "").strip()
        head = " ".join((pg.findtext("running_head") or "").split())
        n = len(lines_of(pg))
        out.append(f"== scan {scan}" + (f" p.{printed}" if printed else "") + f" ({n} lines)"
                   + (f" head: {head[:60]}" if head else ""))
        for c in by_scan[scan]:
            mark = {0: "-", -1: "~", -2: "R"}.get(c["level"], f"H{c['level']}")
            out.append(f"{c['id']} l{c['n']} {mark} {c['text']}")
    return out


def like(title: str, source: str) -> float:
    """How much of the title's letters are in the lines it came from, in order. A label the
    line has lost to the OCR (VIL for CAP. VII.) does not count against it."""
    a, b = folded(title), folded(source)
    if not a:
        return 0.0
    match = lambda x, y: sum(m.size for m in difflib.SequenceMatcher(None, x, y, autojunk=False).get_matching_blocks())
    bare = folded(DIVISION.sub("", title, 1))
    return max(match(a, b) / len(a), match(bare, b) / len(bare) if bare and len(b) <= 12 else 0.0)


def rising(entries: list[dict]) -> list[dict]:
    """The longest run of entries whose first ids go up: one misplaced id costs one entry."""
    best: list[list[dict]] = []
    for e in entries:
        chain = max((c for c in best if c[-1]["ids"][0] < e["ids"][0]), key=len, default=[])
        best.append(chain + [e])
    return max(best, key=len, default=[])


def check(entries: list[dict], cands: list[dict]) -> tuple[list[dict], list[str]]:
    """The entries that hold: known ids, in the book's order, a title that resembles its lines,
    and what was dropped."""
    by_id = {c["id"]: c for c in cands}
    good, notes, used = [], [], set()
    for e in entries:
        ids = sorted({i for i in e["ids"] if i in by_id})
        if not ids:
            notes.append(f"ids: {e}")
            continue
        real = [i for i in ids if by_id[i]["level"] != -2]
        if real and len(real) < len(ids):   # a running head beside the heading it names: the heading counts
            ids = real
        if ids[0] in used:   # two divisions on one line (LIBER II. CAPUT PRIMUM.): the line twice on the sheet
            twin = {**by_id[ids[0]], "id": ids[0] + 0.5}
            by_id[twin["id"]] = twin
            cands.append(twin)
            ids = [twin["id"]]
        used.update(ids)
        src = " ".join(by_id[i]["text"] for i in ids)
        title = " ".join(e["title"].split())
        if like(title, src) < MIN_LIKE and not (len(folded(src)) >= 4 and like(src, title) >= 0.8):   # I N D E X for Index locorum
            notes.append(f"title {title!r} is not in {src!r}")
            continue
        good.append({**e, "ids": ids, "title": title})
    run = rising(good)
    notes += [f"order: {e}" for e in good if e not in run]
    return run, notes


def outline(entries: list[dict], cands: list[dict]) -> list[dict]:
    """The checked entries as written out: the place of their lines, levels from 1."""
    by_id = {c["id"]: c for c in cands}
    out = []
    for e in entries:
        first = by_id[e["ids"][0]]
        out.append({"scan": first["scan"], "line": first["n"], "lines": [[by_id[i]["scan"], by_id[i]["n"]] for i in e["ids"]],
                    "level": min(LEVELS, max(1, int(e["level"]))), "title": e["title"]})
    if out:
        top = min(e["level"] for e in out)
        for e in out:
            e["level"] -= top - 1
    return out


def entry_number(e: dict) -> tuple[str, int] | None:
    """number() of an entry's title, also when the label stands further on (Ideae practicae liber V)."""
    t = e["title"]
    m = BOOK_ANY.search(t)
    return number(t) or (number(t[m.start():]) if m else None)


def line_box(boxes, n: int):
    """The box of line n of a page in pdf_boxes (0-10000 of the page, per line one box or several)."""
    b = boxes[n - 1] if boxes and 0 < n <= len(boxes) else None
    if not b:
        return None
    rows = b if isinstance(b[0], list) else [b]
    return [min(r[0] for r in rows), min(r[1] for r in rows), max(r[2] for r in rows), max(r[3] for r in rows)]


class Scans:
    """Images from the book's PDF, and where its lines are."""

    def __init__(self, folder: Path, xml: Path):
        import pymupdf
        self.pymupdf = pymupdf
        pdf = PDF_DIR / f"{folder.name}.pdf"
        self.doc = pymupdf.open(pdf) if pdf.is_file() else None
        cache = xml.parent / "pdf_boxes.json"
        self.boxes = json.loads(cache.read_text(encoding="utf-8")).get("boxes") or {} if cache.is_file() else {}
        self.lock = threading.Lock()

    def image(self, scan: int, band: tuple[float, float] = (0.0, 1.0), px: int = READ_PX) -> bytes | None:
        """The page, or the band of it between these shares of its height, as JPEG."""
        if self.doc is None or not 0 < scan <= self.doc.page_count:
            return None
        pm = self.pymupdf
        with self.lock:   # PyMuPDF is not thread-safe
            page = self.doc[scan - 1]
            r = page.rect
            clip = pm.Rect(r.x0, r.y0 + band[0] * r.height, r.x1, r.y0 + band[1] * r.height)
            zoom = min(250 / 72, px / max(clip.width, clip.height, 1))
            data = page.get_pixmap(matrix=pm.Matrix(zoom, zoom), clip=clip).tobytes("jpeg", jpg_quality=70)
            del page
            pm.TOOLS.store_shrink(100)   # else every page rendered stays in memory
            return data

    def band(self, scan: int, ns: list[int]) -> tuple[float, float]:
        """The share of the page's height around lines ns; the whole page without boxes."""
        boxes = [b for b in (line_box(self.boxes.get(str(scan)), n) for n in ns) if b]
        if not boxes:
            return (0.0, 1.0)
        y0, y1 = min(b[1] for b in boxes) / 10000, max(b[3] for b in boxes) / 10000
        h = max(0.012, min(b[3] - b[1] for b in boxes) / 10000)
        return (max(0.0, y0 - 3 * h), min(1.0, y1 + 5 * h))

    def close(self):
        if self.doc is not None:
            self.doc.close()


def number(title: str) -> tuple[str, int] | None:
    """(label, number) of a title such as CAP. XIII., Disputatio 4 or Caput primum. Of several
    numbers joined (Regula X, XI, XII, & XIII) the last; a series by chapter (Analysis cap. IX) has
    the first word for its label."""
    m = re.match(r"\W*([^\W\d_]{2,}?)\W+([IVXLCDMivxlcdm]+|\d+)\b", title)
    if m:
        n = m.group(2)
        more = re.match(r"(?:(?:\s*,\s*|[.,]?\s*&\s*|\s+et\s+)(?:[IVXLCDM]+|\d+)\b)+", title[m.end():])
        if more:
            n = re.findall(r"[IVXLCDM]+|\d+", more.group(0))[-1]
        val = int(n) if n.isdigit() else roman(n.upper())
        return (folded(m.group(1))[:3], val) if val else None
    m = re.match(r"\W*([^\W\d_]{2,})\W+([^\W\d_]+)(?:\W+([^\W\d_]+))?", title)
    if m and DIVISION.match(title):
        val = ordinal(folded(m.group(2) + (m.group(3) or "")))
        if val:
            return (folded(m.group(1))[:3], val)
    m = BY_CHAPTER.match(title)
    if m:
        val = int(m.group(2)) if m.group(2).isdigit() else roman(m.group(2).upper())
        return (folded(m.group(1))[:3], val) if val else None
    m = LABEL_LAST.match(title)
    return number(f"{m.group(2)} {m.group(3)}") if m else None


def doubt(outline: list[dict], root, before: int, source: str) -> str:
    """Why this outline should not replace the contents the book had, or "": the signs of the ways
    it has gone wrong without looking wrong. before is the number of entries of the contents made
    without it, source where that came from; an alphabetical register is no measure, so only
    the running heads count for that."""
    scans = sum(1 for _ in root.iter("page"))
    if len(outline) < 2:
        return ""
    if scans >= DOUBT_THICK and source in ("heads", "heads+headings") and before >= DOUBT_MANY and len(outline) < DOUBT_FEWER * before:
        return f"{len(outline)} entries where the running heads had {before}"
    plain = sum(len(r) for r in numbered_runs(outline) if sum(len(e["title"].split()) for e in r) > CHAPTER_WORDS * len(r))
    if plain >= DOUBT_SHARE * len(outline):
        return f"{plain} of {len(outline)} entries are numbered statements"
    return ""


def numbered_runs(outline: list[dict]) -> list[list[dict]]:
    """The series of entries whose title starts with a number, per level: a series ends at an entry
    of its level or higher without a number, and at a numbered one higher up. Statements are judged
    per series, so that many short series of long chapter titles under named parts, and a long series
    of short titles, are no statements."""
    runs, open_ = [], {}

    def close(deeper_than: int):
        for level in [k for k in open_ if k > deeper_than]:
            runs.append(open_.pop(level))

    for e in outline:
        level = int(e["level"])
        if NUMBER_FIRST.match(e["title"]):
            close(level)
            open_.setdefault(level, []).append(e)
        else:
            close(level - 1)
    close(-1)
    return runs


def toc(root) -> list[dict]:
    """The entries of data/<work>/toc_haiku.json of this book, or []."""
    source = root.get("source")
    path = DATA / Path(source).stem / OUT_NAME if source else None
    if path is None or not path.is_file() or (root.get("kind") == "ocr" and not OCR_TOO):
        return []
    return json.loads(path.read_text(encoding="utf-8"))["toc"]


def apply(root, entries: list[dict]):
    """Make the marks in the text say what the outline says: the lines of an entry are headings of
    its level, the first with para="true". In a book that is only OCR the marks were made by a
    rough rule, so every other heading mark goes; a transcription keeps its other headings."""
    chosen = {tuple(x): e["level"] for e in entries for x in e["lines"]}
    firsts = {(e["scan"], e["line"]) for e in entries}
    ocr = root.get("kind") == "ocr"
    for pg in root.iter("page"):
        scan = int(pg.get("n"))
        for l in lines_of(pg):
            at = (scan, int(l.get("n") or 0))
            if l.get("n") and at in chosen:   # line 0 is an entry without a line to mark, not every line without a number
                l.set("type", "heading")
                l.set("level", str(chosen[at]))
                if at in firsts:
                    l.set("para", "true")
                else:
                    l.attrib.pop("para", None)
            elif ocr and l.get("type") == "heading":
                del l.attrib["type"]
                l.attrib.pop("level", None)
