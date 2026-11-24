"""The contents of a Bible commentary: every chapter it treats once, on the scan where it begins, checked
against the canon, and below a chapter the verses or pericopes the book prints as headings.

A model is not trusted to find the chapters. The book itself names them, in its headings, running heads
and margins ("CAP. IV.", "AD ROMANOS. CAP. I. vers. 24.", "Matth. 6. 1.", "PSALMUS XLIII."):
  1. anchors(): every such place as (scan, book, chapter, verse), without a model (bibleref.find_refs and
     the bare labels CAP. N and Vers. N, with the book from the running head or the work's books);
  2. chapters(): per Bible book the longest series of chapters that rises with the scans; a chapter
     starts at its heading, else where the running head first shows it;
  3. gaps: a chapter between two found ones that has no anchor is looked for on the scans between them,
     first in the text (a short line with its number), then by Haiku on the scans (looked());
     what then is still open stays open, as a gap in the result, never a guess;
  4. below a chapter the verse headings (Vers. 23., Ad vers. 35.), the pericopes (cap. 5. a vers. 27. ad
     33.) and the parts of a psalm (Analysis, Scholia), where the book prints them as headings.
The front and back matter (dedication, preface, indexes) come from toc_context's outline of the book.

The works are those genre_haiku took for a commentary on the Bible (bijbel = commentaar) with the
titled work in the scan, transcribed. The result is data/<work>/toc_commentary.json, or with --out
elsewhere, for a trial. Questions only to Haiku, through the claude command line (toc_context.ask).

Examples
  python commentary.py W01521 --out ~/.cache/toc_context/n09/comm/run --show
  python commentary.py --list test.txt --out DIR
  python commentary.py --works                        # the commentaries that can be done
"""

import argparse
import base64
import bisect
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from references import bibleref as B  # noqa: E402
from contents import toc_context as C  # noqa: E402
from common.paths import PROMPTS, genre_answers  # noqa: E402
from common.works import id_of  # noqa: E402
from contents import headings as T  # noqa: E402  (was toc_haiku: REFACTOR_PLAN.md steps 36 and 37)

GENRES = genre_answers()   # genre_haiku.CACHE, with the books
OUT_NAME = "toc_commentary.json"
RAW = "toc_commentary/raw"
SCAN_PROMPT = PROMPTS / "commentary_scan.txt"
SHORT = 90            # characters of a line that can be a heading without being marked as one
TOP = 6               # lines at the top of a page where an unmarked chapter heading is looked for
LOOK_SCANS = 6        # scans in one question to Haiku
LOOK_ROUNDS = 3       # questions at most for one chapter (18 scans)
BOOK_PAGES = 3        # pages that must name a book for it to count as treated
RH_CHAPTERS = 3       # chapters of a book the running heads must name for the book to count as treated
SELECTIVE = 0.6       # share of the chapters between its first and last found below which a book is treated selectively: its gaps are no misses
RH_AHEAD = 3          # scans after a running head's first show of a chapter where its heading can still come
RH_BACK = 12          # scans before a running head's first show of a chapter where Haiku looks for its heading
NEAR = 1              # scans before a running head's first show of a chapter where its heading may stand
LATIN = {
    "Genesis": "Genesis", "Exodus": "Exodus", "Leviticus": "Leviticus", "Numeri": "Numeri", "Deuteronomium": "Deuteronomium",
    "Jozua": "Iosua", "Richteren": "Iudices", "Ruth": "Ruth", "1 Samuel": "I Samuelis", "2 Samuel": "II Samuelis",
    "1 Koningen": "I Regum", "2 Koningen": "II Regum", "1 Kronieken": "I Chronicorum", "2 Kronieken": "II Chronicorum",
    "Ezra": "Esdras", "Nehemia": "Nehemias", "Esther": "Esther", "Job": "Iob", "Psalmen": "Psalmi", "Spreuken": "Proverbia",
    "Prediker": "Ecclesiastes", "Hooglied": "Canticum canticorum", "Jesaja": "Iesaias", "Jeremia": "Ieremias",
    "Klaagliederen": "Threni", "Ezechiel": "Ezechiel", "Daniel": "Daniel", "Hosea": "Hosea", "Joel": "Ioel", "Amos": "Amos",
    "Obadja": "Obadias", "Jona": "Ionas", "Micha": "Michaeas", "Nahum": "Nahum", "Habakuk": "Habacuc", "Sefanja": "Sophonias",
    "Haggai": "Haggaeus", "Zacharia": "Zacharias", "Maleachi": "Malachias", "Mattheus": "Matthaeus", "Marcus": "Marcus",
    "Lucas": "Lucas", "Johannes": "Iohannes", "Handelingen": "Acta apostolorum", "Romeinen": "Ad Romanos",
    "1 Korintiers": "I ad Corinthios", "2 Korintiers": "II ad Corinthios", "Galaten": "Ad Galatas", "Efeziers": "Ad Ephesios",
    "Filippenzen": "Ad Philippenses", "Kolossenzen": "Ad Colossenses", "1 Tessalonicenzen": "I ad Thessalonicenses",
    "2 Tessalonicenzen": "II ad Thessalonicenses", "1 Timotheus": "I ad Timotheum", "2 Timotheus": "II ad Timotheum",
    "Titus": "Ad Titum", "Filemon": "Ad Philemonem", "Hebreeen": "Ad Hebraeos", "Jakobus": "Iacobi", "1 Petrus": "I Petri",
    "2 Petrus": "II Petri", "1 Johannes": "I Iohannis", "2 Johannes": "II Iohannis", "3 Johannes": "III Iohannis",
    "Judas": "Iudae", "Openbaring": "Apocalypsis",
}
NUM = r"([IVXLCDMJ]{1,8}|\d{1,3})"
PSALM = re.compile(r"(?:^|\W)psal(?:mus|mi|mo|m\.|\.)?\s*\.?\s*" + NUM + r"\b", re.I)
VERSE = re.compile(r"^\W{0,3}(?:ad\s+)?(?:vers(?:us|u|ic\w*|\.)?|verf\.?|v\.|ψ\.?)\s*(\d{1,3})\b(?:\s*(?:[-–,&]|et|ad)\s*(\d{1,3}))?", re.I)
RANGE = re.compile(r"(?:à|a|ab)\s*(?:vers(?:u|\.)?|verf\.?|v\.)\s*(\d{1,3})\.?\s*(?:ad|usque ad|-|–)\s*(?:vers(?:um|\.)?|v\.)?\s*(\d{1,3})", re.I)
ORDINAL_AFTER = re.compile(r"\b((?:ad\s+)?[A-Za-z]{4,})[.,]?\s+(I{1,3})\b\.?(?=\s*(?:$|cap))", re.I)   # AD CORINTHIOS I., PETRI II.
UNITS = {"prim": 1, "secund": 2, "terti": 3, "quart": 4, "quint": 5, "sext": 6, "septim": 7, "octau": 8, "non": 9,
         "undecim": 11, "duodecim": 12}
TENS = {"decim": 10, "uicesim": 20, "uigesim": 20, "tricesim": 30, "trigesim": 30, "quadragesim": 40, "quinquagesim": 50,
        "sexagesim": 60, "septuagesim": 70, "octogesim": 80, "nonagesim": 90, "centesim": 100}
ORD_STEM = "|".join(sorted(list(UNITS) + list(TENS), key=len, reverse=True))
ORD = rf"((?:(?:{ORD_STEM})(?:um|us|i|o|a|ae|am|is)?\s*(?:et\s*)?){{1,3}})"   # on norm()'d text: v is u
CAP_LABEL = r"(?:cap(?:ut|itulum|itis|ite|it+el|\.)?|kapit+el|chap(?:ter)?|hooftst[uv]ck)(?![a-z])"   # not CAPITE+L: Das XIX. Capitel
CAP_NUM = re.compile(r"(?:^|[\s.,;:(])" + CAP_LABEL + r"\s*\.?\s*(?:([iuxlcdm]{1,8}|\d{1,3})\b|" + ORD + r"(?![a-z]))")
NUM_CAP = re.compile(r"(?:^|[\s.,;:(])" + ORD + r"\s+" + CAP_LABEL + r"\b")   # ARGVMENTVM PRIMI CAPITIS
VERNACULAR = {"erst": 1, "eerst": 1, "first": 1, "ander": 2, "tweed": 2, "second": 2, "dritt": 3, "derd": 3, "third": 3,
              "uiert": 4, "uierd": 4, "fourth": 4, "funft": 5, "fünft": 5, "uijfd": 5, "fifth": 5, "sechst": 6, "sesd": 6,
              "zesd": 6, "sixth": 6, "sibend": 7, "siebend": 7, "seuend": 7, "zeuend": 7, "seuenth": 7, "acht": 8, "eighth": 8,
              "neund": 9, "negend": 9, "ninth": 9, "zehend": 10, "tiend": 10, "thiend": 10, "tenth": 10}
NUM_LABEL = re.compile(r"(?:^|[\s.,;:(])(?:das|dz|het|the)?\s*(?:([iuxlc]{1,8}|\d{1,3})\.?|(" + "|".join(VERNACULAR)
                       + r")[a-z]{0,3})\s+(?:cap(?:it+el)|kapit+el|chapter|hooftst[uv]ck)\b")   # Das XIX. Capitel, Das erst Capitel
LABEL_LINE = 45       # characters of a line at most that is a heading with its label not at the start
LABEL_NEAR = 25       # characters from the start of a line within which its chapter label makes it a heading
PSALM_PART = re.compile(r"^\W*([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)?)\W+(?:in|ex|ad|psal)", re.I)


def roman(n: int) -> str:
    out = ""
    for v, s in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                 (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + s, n - v
    return out


def number(s: str) -> int:
    return B.number(B.norm(s).replace("j", "i"))


def ordinal(s: str) -> int:
    """decimumseptimum, septimum decimum, vicesimum primum: 17, 17, 21."""
    n = 0
    for m in re.finditer(ORD_STEM, B.norm(s)):
        n += UNITS.get(m.group(0)) or TENS.get(m.group(0), 0)
    return n


def chapter_in(t: str) -> tuple[int, int, int] | None:
    """(where, chapter, end) of the first chapter label with its number in t: CAP. IV., CAPVT 4, Caput primum,
    CAPITIS OCTAVI, PRIMI CAPITIS."""
    n = B.norm(t)
    found = []
    for m in CAP_NUM.finditer(n):
        found.append((m.start(), number(m.group(1)) if m.group(1) else ordinal(m.group(2)), m.end()))
        break
    for m in NUM_CAP.finditer(n):
        found.append((m.start(), ordinal(m.group(1)), m.end()))
        break
    for m in NUM_LABEL.finditer(n):
        found.append((m.start(), number(m.group(1)) if m.group(1) else VERNACULAR[m.group(2)], m.end()))
        break
    found = [f for f in found if f[1]]
    return min(found) if found else None


def labelled(t: str) -> bool:
    """A line that begins with a chapter label (or has one within LABEL_NEAR characters: ANALYSIS CAP. IV.)."""
    at = chapter_in(t)
    if not at:
        return bool(PSALM.match(t))
    short = len(t) <= LABEL_LINE
    stops = t[at[2]:at[2] + 2].lstrip().startswith((".", ":", ",")) or not t[at[2]:].strip()   # CAP. IV. De fide: no sentence running on
    lead = not re.search(r"\d", t[:at[0]])   # not 28.cap.2.vers.3.: the end of a citation
    return lead and (at[0] <= 3 and (short or stops) or at[0] <= LABEL_NEAR and short)


def works() -> dict[str, dict]:
    """{work: genre_haiku's answer} of the transcribed commentaries."""
    genres = json.loads(GENRES.read_text(encoding="utf-8"))
    out = {}
    for w, g in genres.items():
        if g.get("bijbel") == "commentaar" and g.get("match") == "ja" and g.get("bb"):
            found = sorted(T.DATA.glob(f"{w}_*/book.xml"))
            if found:
                out[w] = g
    return out


def text(el) -> str:
    return " ".join(T.R.plain(el).split())


SPREAD_ORDINAL = re.compile(r"^\W*(?:in\s+)?(I{1,3})\.?\s+epist", re.I)   # IN I. EPIST. D. PAVLI facing AD TIMOTHEVM CAP. I.


def spread(head: str, facing: str) -> str:
    """A running head read with the facing page's when it names a chapter but not its book: the book on the
    left page and the chapter on the right (GENESEOS. | CAPUT II.), or the epistle's number on the left
    (IN II. EPIST. D. PAVLI | AD TIMOTHEVM CAP. I.)."""
    if not head or not facing or not chapter_in(head) or B.find_refs(head) or named_book(head, B.BOOKS, whole=True):
        return head
    m = SPREAD_ORDINAL.match(facing)
    if m and named_book(f"{m.group(1)}. {head}", B.BOOKS, whole=True):
        return f"{m.group(1)}. {head}"
    if named_book(facing, B.BOOKS, whole=True) and not chapter_in(facing) and not B.find_refs(facing):
        return f"{facing} {head}"
    return head


def pages(root):
    """Per page: scan, running head (read with the facing page's, spread()), [(line number, text, marked as
    heading)] with the <head> blocks, margins."""
    facing = (-1, "")
    for pg in root.iter("page"):
        lines = [(int(l.get("n") or 0), text(l), l.get("type") == "heading") for l in T.lines_of(pg)]
        lines += [(len(lines) + k, text(h), True) for k, h in enumerate(pg.findall("head"), 1)]
        margins = [text(n) for m in pg.iter("marginalia") for n in m]
        scan = int(pg.get("n"))
        head = text(pg.find("running_head")) if pg.find("running_head") is not None else ""
        read = spread(head, facing[1]) if facing[0] == scan - 1 else head
        facing = (scan, head)
        yield scan, read, lines, margins


AUTHOR = [""]   # the stem of the author's surname of the work at hand (run() sets it): DANIELIS TOSSANI is no book


NUMBERED = {"prim": "I", "secund": "II", "terti": "III", "quart": "IIII", "prior": "I", "posterior": "II"}
NUM_WORD = r"(prim|sec[uv]nd|terti|q[uv]art|prior|posterior)(?:[uv]s|[uv]m|am|ae|em|is|e|i|o|a)?\b"   # AD TIMOTHEVM POSTERIOREM
NOT_NAME = r"(?!cap|chap|epist|lib|homil|lect|serm|pars|part|tom|concio|quaest|psal)"   # CAPVT PRIMVM, EPISTOLA PRIMA
num_word = lambda w: NUMBERED[w.lower().replace("v", "u").replace("ſ", "s")]   # SECVNDVS, ſecundus


def numbered(m, num: int, name: int) -> str:
    """The number before the name, if the name is a numbered Bible book's: not ARGVMENTVM PRIMI CAPITIS."""
    word = B.norm(m.group(name).lower())
    if not any(isinstance(books, dict) and re.fullmatch(rf"h?(?:{stem})[a-z]{{0,3}}", word) for stem, books in B.NAMES):
        return m.group(0)
    return f"{num_word(m.group(num))} {m.group(name)}"


NUMBERED_FORMS = [
    (re.compile(r"\bmalachim\b", re.I), "REGVM"),   # the Hebrew name of Kings, not Malachi (MALACHIM PRIMVS: 1 Kings)
    (re.compile(r"\b(I{1,3}|IV|IIII|[1-4])\.?\s+libr(?:um|vm|i)\s+", re.I), r"\1 "),   # IN II. LIBRVM SAMVELIS
    (re.compile(r"\bliber\s+" + NUM_WORD + r"\s+" + NOT_NAME + r"([a-z]{4,})", re.I), lambda m: numbered(m, 1, 2)),   # LIBER TERTIVS REGVM
    (re.compile(r"\b" + NUM_WORD + r"\s+(?:liber|libr[uv]m|epist[a-z]*)\s+(?:[ds]\.\s*)?" + NOT_NAME + r"([a-z]{4,})", re.I), lambda m: numbered(m, 1, 2)),   # QVARTVS LIBER REGVM, PRIMAM EPISTOLAM D. PETRI
    (re.compile(r"\b(?!sec[uv]nd[uv]m)" + NUM_WORD + r"\s+" + NOT_NAME + r"([a-z]{4,})", re.I), lambda m: numbered(m, 1, 2)),   # PRIMAM IOANNIS; not EVANGELIVM SECVNDVM IOANNEM
    # the number last, or before a chapter: REGVM, SECVNDVM HEBRAEOS is "according to the Hebrews"
    (re.compile(r"\b" + NOT_NAME + r"([a-z]{4,})\.?,?\s+(?:liber\s+)?" + NUM_WORD + r"(?=\W*(?:$|[/|]|cap|lib)|\s*,)", re.I), lambda m: numbered(m, 2, 1)),   # SAMVELIS (LIBER) PRIMVS, PARALIP. SECVNDVS
]


def numbered_form(t: str) -> str:
    """A numbered book's number put before its name, as bibleref reads it: IN II. LIBRVM SAMVELIS, SAMVELIS
    PRIMVS, ESDRAE LIBER SECVNDVS, LIBER TERTIVS REGVM."""
    for pattern, to in NUMBERED_FORMS:
        t = pattern.sub(to, t)
    return t


WHOLE_SHORT = {"ruth", "iob", "ioel", "amos", "ionas"}   # LIBER RVTH. CAPVT PRIMVM.: a whole name, no abbreviation


def named_book(t: str, books: list[str], whole: bool = False) -> str | None:
    """One of the work's books named in t without a number (AD ROMANOS, MATTHAEI), the longest name if more;
    whole: no short abbreviation followed by more words (IOAN. VVOLPHII is the author's name)."""
    t = ORDINAL_AFTER.sub(lambda m: f"{m.group(2)} {m.group(1)}", numbered_form(t).replace("æ", "ae").replace("Æ", "AE").replace("œ", "oe").replace("Œ", "OE"))
    n = B.norm(t)
    abbreviated = lambda m: (len(m.group(0)) <= 4 and m.group(0).lstrip("h") not in WHOLE_SHORT
                             and n[m.end():m.end() + 1] == "." and n[m.end() + 1:].strip(" .,;:0-9iuxlc"))
    hits = []
    for b in books:
        exact, loose = name_pattern(b)
        m = re.search(exact, n) or re.search(loose, n)
        core = len(m.group(0)) - len(m.group("end")) if m else 0
        if m and AUTHOR[0] and n[m.end():].lstrip(" .,").startswith(AUTHOR[0]):
            continue
        if m and core >= 3 and not (whole and abbreviated(m)) and (not m.group("end") or core >= 4):
            hits.append((-len(m.group(0)), m.start(), b))
    return min(hits)[2] if hits else None


PATTERNS = {}


def name_pattern(book: str):
    """bibleref's names of a book, also with a Latin ending the list lacks (APOCALYPSIN, ECCLESIASTEN)."""
    if book not in PATTERNS:
        p = B.book_pattern(book)
        p = p.replace("(?<![a-z0-9])(?:", "(?<![a-z0-9])h?(?:", 1)   # HESTHERAE, HIEREMIAS
        PATTERNS[book] = (re.compile(p.removesuffix("(?![a-z])") + "(?P<end>)(?![a-z])"),
                          re.compile(p.removesuffix("(?![a-z])") + "(?P<end>[a-z]{0,3})(?![a-z])"))
    return PATTERNS[book]


def refs_of(t: str) -> list[tuple]:
    """bibleref.find_refs, but a reference whose line names another book before it belongs to that book:
    in APOCALYPSIS S. IOHANNIS CAP. XII. the saint's name is the author's, the chapter is Revelation's."""
    out = []
    for b, c, v, start, end in B.find_refs(t):
        before = named_book(t[:start], B.BOOKS, whole=True)
        if before and before != b and before == named_book(t, B.BOOKS, whole=True) and c <= B.CHAPTERS[before] and not B.find_refs(t[:start]):
            b = before
        out.append((b, c, v, start, end))
    return out


def parse(t: str, books: list[str], current: str | None, whole: bool = False) -> list[tuple[str, int, int]]:
    """(book, chapter, verse) named in t: full references to one of the work's books, or a bare CAP. N
    or PSALMUS N with the book named beside it, else the current one. Verse 0 is the chapter."""
    out = [(b, c, v) for b, c, v, *_ in refs_of(t) if b in books]
    if out:
        return out
    m = PSALM.search(t)
    if m and "Psalmen" in books:
        return [("Psalmen", number(m.group(1)), 0)]
    at = chapter_in(numbered_form(t))   # PETRI SECVNDA CAP. 3: the epistle's number is no chapter
    if at:
        book = named_book(t, books, whole) or current
        if book:
            return [(book, at[1], 0)]
    return []


LISTED = re.compile(r"\s*(?:[.,;]|&|et)?\s*(?:&|et)?\s*([IVXLC]{1,7}|\d{1,3})\b\.?", re.I)


def listed(t: str, at: tuple[int, int, int]) -> list[int]:
    """The chapter chapter_in() found and those that follow it in a list: CAPUT VII. VIII. & XI."""
    out, end = [at[1]], at[2]
    while m := LISTED.match(t, end):
        out.append(number(m.group(1)))
        end = m.end()
    return out


def evidence(root) -> dict[str, dict]:
    """Per Bible book what the book itself shows of it: {"rh": pages whose running head names it, "titles":
    title headings of its own, "chs": its chapters named with it, "first": the first scan that names it, "last":
    the last}. See treated()."""
    rh, titles, chs, first, last = {}, {}, {}, {}, {}
    for scan, head, lines, _ in pages(root):
        named = set()
        book = named_book(head, B.BOOKS, whole=True)
        refs = [(b, c) for b, c, *_ in refs_of(head)]
        at = chapter_in(head)
        if not refs and at and book:
            refs = [(book, c) for c in listed(head, at)]   # NEHEMIÆ. CAPUT VIII. & IX.
        for b, c in refs[:1] + [r for r in refs[1:] if r[0] == refs[0][0]]:
            if 1 <= c <= B.CHAPTERS[b]:
                chs.setdefault(b, set()).add(c)
                named.add(b)
        if book:
            named.add(book)
        for b in named:
            rh[b] = rh.get(b, 0) + 1
            first.setdefault(b, scan)
            last[b] = scan
        for k, (n, t, marked) in enumerate(lines):
            # an unmarked line counts only as a short labelled one at the top of a page: lower down, a line
            # that starts with a chapter cites it (testatur & Johannes cap. 1. v. 6. in a commentary on Matthew)
            if not t or len(t) > SHORT or not (marked or labelled(t) and k < TOP and len(t) <= LABEL_LINE):
                continue
            book = named_book(t, B.BOOKS, whole=True)
            if not book:
                continue
            at = chapter_in(t)
            if at:
                chs.setdefault(book, set()).add(at[1])
                last[book] = max(last.get(book, scan), scan)
            elif marked and not B.find_refs(t) and len(t) <= LABEL_LINE:
                titles[book] = titles.get(book, 0) + 1
                first.setdefault(book, scan)
                last[book] = max(last.get(book, scan), scan)
    return {b: {"rh": rh.get(b, 0), "titles": titles.get(b, 0), "chs": chs.get(b, set()), "first": first[b],
                "last": last.get(b, first[b])} for b in first}


def treated(root, hint: list[str], ev: dict | None = None) -> list[str]:
    """The Bible books the commentary treats, in the order it treats them, from the book itself:
    - a book the running heads name on BOOK_PAGES pages or more, with RH_CHAPTERS of its chapters named (a
      book of one chapter needs none), or
      on BOOK_PAGES * 3 pages (ECCLESIASTES, JOELIS);
    - a book with a title heading of its own (IOELIS PROPHETAE) and RH_CHAPTERS chapters that name it.
    Chapters count as evidence only where the book is named with them. A margin never counts: it cites; a
    line of text neither. genre_haiku's books are only a hint, used when the book itself names none."""
    ev = evidence(root) if ev is None else ev
    rh = {b: e["rh"] for b, e in ev.items()}
    titles = {b: e["titles"] for b, e in ev.items()}
    chs = {b: e["chs"] for b, e in ev.items()}
    first = {b: e["first"] for b, e in ev.items()}
    enough = lambda b: B.CHAPTERS[b] == 1 or len(chs.get(b, ())) >= min(RH_CHAPTERS, B.CHAPTERS[b] - 1)   # AD PHILEMONEM.
    books = [b for b in first if rh.get(b, 0) >= BOOK_PAGES and (enough(b) or rh[b] >= BOOK_PAGES * 3)
             or titles.get(b) and len(chs.get(b, ())) >= RH_CHAPTERS]
    return sorted(books, key=lambda b: first[b]) or hint


def bare_ref(t: str) -> bool:
    """A line that is a reference and little else (Matth. 6. 1., 2. Petr. 2.): a lemma heading, not a citation."""
    refs = B.find_refs(t)
    return bool(refs) and refs[0][3] <= 1 and len(t) <= 30


def anchors(root, books: list[str]) -> list[dict]:
    """Every place where the book names a chapter of one of its books: [{scan, line, kind (head: a heading
    line, top: a short line at the top of a page, rh: running head, margin), book, ch, v, text}], in the
    order of the book; verse headings without a chapter take the chapter of the place before them."""
    out, current, chapter = [], books[0] if len(books) == 1 else None, None
    for scan, head, lines, margins in pages(root):
        if head:
            current = named_book(head, books, whole=True) or current   # a spread can name the book on one page only
            for b, c, v in parse(head, books, current)[:1]:
                if 1 <= c <= B.CHAPTERS[b]:
                    out.append({"scan": scan, "line": 0, "kind": "rh", "book": b, "ch": c, "v": v, "text": head})
            current = named_book(head, books) or current
        for k, (n, t, marked) in enumerate(lines):
            if not t or len(t) > (200 if marked else SHORT):
                continue
            label = labelled(t) or bare_ref(t)
            if not marked and k >= TOP and not VERSE.match(t) and not label:
                continue
            title = t[:SHORT] if marked else t   # IN EVANGELIVM ... SECVNDVM MATTHAEVM RODOLPHI GVALTHERI ... (a long title)
            if (marked or k < TOP) and len(title) <= SHORT and not chapter_in(title) and not B.find_refs(title):
                current = named_book(title, books, whole=True) or current   # a book's title: IOELIS PROPHETAE
            found = parse(t, books, current)
            if found and not marked and current and found[0][0] != current and not named_book(t, books, whole=True):
                found = []   # an unmarked line naming another book's chapter is a citation (Joel 2. 28. on a page of Hosea)
            if found and (marked and (label or len(t) <= SHORT) or label):
                b, c, v = found[0]
                if 1 <= c <= B.CHAPTERS[b]:
                    out.append({"scan": scan, "line": n, "kind": "head" if marked else "top", "book": b, "ch": c, "v": v, "text": t})
                    current, chapter = b, (b, c)
                continue
            m = VERSE.match(t)
            if m and chapter and (marked or k < TOP or len(t) < 60):
                out.append({"scan": scan, "line": n, "kind": "verse", "book": chapter[0], "ch": chapter[1], "v": int(m.group(1)),
                            "to": int(m.group(2) or 0), "text": t})
        for t in margins:   # a margin names the chapter of the book at hand, never another book: that is a citation
            for b, c, v in parse(t, [current] if current else [], current)[:1]:
                if 1 <= c <= B.CHAPTERS[b] and chapter_in(t):
                    out.append({"scan": scan, "line": 0, "kind": "margin", "book": b, "ch": c, "v": v, "text": t})
        if out and out[-1]["kind"] in ("rh", "head", "top"):
            chapter = (out[-1]["book"], out[-1]["ch"])
    return out


def rising(items: list[tuple[int, int]]) -> list[int]:
    """Indexes of the longest series of strictly rising chapter numbers (items: (chapter, index))."""
    tails, prev, at = [], [-1] * len(items), []
    for i, (c, _) in enumerate(items):
        k = bisect.bisect_left([items[j][0] for j in tails], c)
        if k == len(tails):
            tails.append(i)
        else:
            tails[k] = i
        prev[i] = tails[k - 1] if k else -1
    i = tails[-1] if tails else -1
    while i >= 0:
        at.append(i)
        i = prev[i]
    return at[::-1]


COMMENT_ON = re.compile(r"^\W*(?:in|ad)\s+" + CAP_LABEL)   # IN CAPVT PRIMVM, AD CAP. III. (not SCHOLIA IN CAP.: that follows the start)


def chapters(found: list[dict], books: list[str]) -> tuple[list[dict], list[dict]]:
    """The chapters, one each, where they begin: [{book, ch, scan, line, how, text}], and the gaps between
    them: [{book, ch, lo, hi}] (the scans between the chapters around it)."""
    out, gaps = [], []
    for b in books:
        mine = [a for a in found if a["book"] == b and (a["kind"] == "rh" or a["kind"] == "margin" and not a["v"]
                                                        or a["kind"] in ("head", "top") and a["v"] in (0, 1))]
        first = {}
        for a in mine:   # the first place each chapter is named, a heading before a running head on the same scan
            if a["ch"] not in first or (a["scan"] < first[a["ch"]]["scan"]):
                first[a["ch"]] = a
        shown = {}
        for a in mine:
            if a["kind"] == "rh" and a["ch"] not in shown:
                shown[a["ch"]] = a
        start = {}
        # IN CAPVT PRIMVM: the commentary on a chapter, where the book prints its text with CAPUT I. before it (Drusius)
        on = {x["ch"] for x in mine if x["kind"] in ("head", "top") and COMMENT_ON.match(B.norm(x["text"]))}
        for c in first:
            hs = [x for x in mine if x["ch"] == c and x["kind"] in ("head", "top")]
            if len(on) >= 2 and c in on:
                hs = [x for x in hs if COMMENT_ON.match(B.norm(x["text"]))]
            if c in shown:   # the running head says where the chapter is: its heading stands there or before, after the chapter before
                r = shown[c]["scan"]
                since = shown[c - 1]["scan"] if c - 1 in shown and shown[c - 1]["scan"] < r else r - NEAR - 1
                near = [x for x in hs if since <= x["scan"] <= r + RH_AHEAD]   # a running head can show a chapter before its heading
                nxt = min((x["scan"] for x in mine if x["ch"] == c + 1 and x["kind"] in ("head", "top")), default=10 ** 6)
                later = [x for x in hs if r + RH_AHEAD < x["scan"] < nxt]   # a misprinted running head: the heading comes later
                start[c] = near[0] if near else later[0] if later else shown[c]
            else:
                start[c] = hs[0] if hs else first[c]
        series = sorted(start.values(), key=lambda a: (a["scan"], a["line"]))
        keep = [series[i] for i in rising([(a["ch"], i) for i, a in enumerate(series)])]
        for a in keep:
            out.append({"book": b, "ch": a["ch"], "scan": a["scan"], "line": a["line"], "how": a["kind"], "text": a["text"]})
        for x, y in zip(keep, keep[1:]):
            for c in range(x["ch"] + 1, y["ch"]):
                gaps.append({"book": b, "ch": c, "lo": x["scan"], "hi": y["scan"]})
    return sorted(out, key=lambda e: (e["scan"], e["line"])), gaps


def in_text(gap: dict, root) -> dict | None:
    """A chapter of a gap found in the text: a line between its neighbours that is short and carries its
    number behind CAP or alone (CAP. XXXII., XXXII.), one only."""
    want = gap["ch"]
    hits = []
    for scan, _, lines, _ in pages(root):
        if not gap["lo"] <= scan <= gap["hi"]:
            continue
        for n, t, marked in lines:
            if len(t) > SHORT:
                continue
            at = chapter_in(t)
            m = re.match(r"^\W*" + NUM + r"\W*$", t)
            if (at and at[1] == want) or (m and number(m.group(1)) == want):
                hits.append({"scan": scan, "line": n, "text": t})
    return hits[0] if len(hits) == 1 else None


def looked(gap: dict, scans, raw: Path, title: str) -> tuple[dict | None, float]:
    """A chapter of a gap sought by Haiku on the scans between its neighbours, LOOK_SCANS at a time."""
    if scans.doc is None:
        return None, 0.0
    cost = 0.0
    name = f"{LATIN[gap['book']]}, caput {roman(gap['ch'])} ({gap['ch']})"
    for k in range(LOOK_ROUNDS):
        lo = gap["lo"] + k * LOOK_SCANS
        if lo > gap["hi"]:
            break
        shown = list(range(lo, min(gap["hi"], lo + LOOK_SCANS - 1) + 1))
        content = [{"type": "text", "text": f"Book: {title}\nThe chapter sought: {name}\n"}]
        for n in shown:
            image = scans.image(n, px=C.LOOK_PX)
            if image:
                content += [{"type": "text", "text": f"Scan {n}:"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image).decode()}}]
        reply = C.ask(SCAN_PROMPT, content, raw, f"{title}\0{name}\0{shown}")
        cost += reply["cost"]
        a = reply["answer"]
        if a.get("found") and a.get("scan") in shown:
            return {"scan": a["scan"], "line": 0, "text": " ".join(str(a.get("heading") or "").split())}, cost
    return None, cost


def title_of(label: str, t: str) -> str:
    """The label and what the heading says besides its number: Cap. III. De fide."""
    rest = re.sub(r"^.*?\b(?:cap(?:ut|vt|itulum|\.)?|psal(?:mus|m\.|\.)?)\s*\.?\s*" + NUM + r"\b\W*", "", t, count=1, flags=re.I)
    rest = rest if rest != t else ""
    rest = re.sub(r"^\W+", "", rest)
    if B.find_refs(rest) and len(rest) < 25:
        rest = ""
    return f"{label}. {rest}".strip() if rest else label


def below(found: list[dict], chs: list[dict]) -> list[dict]:
    """The verses, pericopes and psalm parts printed as headings, under their chapter."""
    out = []
    starts = [(c["scan"], c["line"]) for c in chs]
    for a in found:
        if a["kind"] == "verse" or (a["kind"] in ("head", "top") and a["v"]):
            k = bisect.bisect_right(starts, (a["scan"], a["line"])) - 1
            if k < 0 or (chs[k]["book"], chs[k]["ch"]) != (a["book"], a["ch"]):
                continue
            r = RANGE.search(a["text"])
            if r:
                label = f"Cap. {roman(a['ch'])}, v. {r.group(1)}–{r.group(2)}"
                rest = a["text"][r.end():]
            else:
                v, to = a["v"], a.get("to") or 0
                label = f"v. {v}" + (f"–{to}" if to > v else "")
                m = VERSE.match(a["text"])
                rest = a["text"][m.end():] if m else ""
            rest = re.sub(r"^\W+", "", rest)
            out.append({"scan": a["scan"], "line": a["line"], "book": a["book"], "ch": a["ch"], "v": a["v"], "kind": "verse",
                        "title": f"{label}. {rest}".strip() if rest else label})
    seen, kept = set(), []
    for e in out:   # one entry per verse heading, the first
        key = (e["book"], e["ch"], e["title"][:20])
        if key not in seen:
            seen.add(key)
            kept.append(e)
    return kept


def matter(work: str, folder: Path, first: int, last: int) -> list[dict]:
    """Front and back matter from toc_context's outline: its entries before the first chapter, and those after
    the last that are no part of the commentary (index, errata, appendix)."""
    path = folder / C.OUT_NAME
    if not path.is_file():
        return []
    toc = json.loads(path.read_text(encoding="utf-8"))["toc"]
    return [{"scan": e["scan"], "line": e["line"], "level": 1, "title": e["title"], "kind": "matter"}
            for e in toc if e["scan"] < first or (e["scan"] > last and T.BACK.match(e["title"]))]


def run(work: str, out: Path | None = None, context: Path | None = None, ask: bool = True) -> dict:
    folder = T.folder_of(work)
    xml = folder / "book.xml"
    root = etree.parse(str(xml), etree.XMLParser(recover=True)).getroot()
    g = works().get(work) or json.loads(GENRES.read_text(encoding="utf-8"))[work]
    AUTHOR[0] = B.norm(folder.name.split("_")[1])[:5] if folder.name.count("_") >= 2 else ""
    books = treated(root, [b for b in g["bb"] if b in B.CHAPTERS])
    found = anchors(root, books)
    chs, gaps = chapters(found, books)
    raw = out / "raw" if out else folder / RAW
    raw.mkdir(parents=True, exist_ok=True)
    scans = T.Scans(folder, xml)
    cost, still = 0.0, []
    span = {}
    for c in chs:
        lo, hi, n = span.get(c["book"], (c["ch"], c["ch"], 0))
        span[c["book"]] = (min(lo, c["ch"]), max(hi, c["ch"]), n + 1)
    selective = {b for b, (lo, hi, n) in span.items() if n < SELECTIVE * (hi - lo + 1)}
    for gap in gaps:
        gap["selective"] = gap["book"] in selective
        hit, how = in_text(gap, root), "text"
        if not hit and ask and not gap["selective"]:
            hit, c = looked(gap, scans, raw, root.get("source", folder.name))
            cost, how = cost + c, "scan"
        if hit:
            chs.append({"book": gap["book"], "ch": gap["ch"], "scan": hit["scan"], "line": hit["line"], "how": how, "text": hit["text"]})
        else:
            still.append(gap)
    chs.sort(key=lambda e: (e["scan"], e["line"]))
    if ask:   # a chapter only the running head shows: its heading is looked for on the scans before it
        for k, c in enumerate(chs):
            if c["how"] != "rh":
                continue
            lo = max(chs[k - 1]["scan"] if k else 1, c["scan"] - RH_BACK)
            hi = min(c["scan"] + RH_AHEAD, chs[k + 1]["scan"] if k + 1 < len(chs) else c["scan"] + RH_AHEAD)
            hit, cc = looked({"book": c["book"], "ch": c["ch"], "lo": lo, "hi": hi}, scans, raw, root.get("source", folder.name))
            cost += cc
            if hit:
                c.update(scan=hit["scan"], line=0, how="rh+scan", text=hit["text"])
        chs.sort(key=lambda e: (e["scan"], e["line"]))
    scans.close()
    many = len({c["book"] for c in chs}) > 1
    label = lambda c: f"Psalmus {roman(c['ch'])}" if c["book"] == "Psalmen" else f"Cap. {roman(c['ch'])}"
    entries = []
    for b in dict.fromkeys(c["book"] for c in chs):
        if many:
            first = min((c for c in chs if c["book"] == b), key=lambda c: c["scan"])
            entries.append({"scan": first["scan"], "line": first["line"], "level": 1, "title": LATIN[b], "kind": "book", "book": b})
    for c in chs:
        own = c["text"] if c["how"] in ("head", "top", "text", "scan", "rh+scan") else ""   # a running head is no title
        entries.append({"scan": c["scan"], "line": c["line"], "level": 2 if many else 1, "title": title_of(label(c), own),
                        "kind": "chapter", "book": c["book"], "ch": c["ch"], "how": c["how"]})
    for v in below(found, chs):
        entries.append({**v, "level": 3 if many else 2})
    if chs:
        ctx = (context / f"{work}.json") if context else None
        if ctx and ctx.is_file():
            toc = json.loads(ctx.read_text(encoding="utf-8"))["toc"]
            first, last = chs[0]["scan"], chs[-1]["scan"]
            entries += [{"scan": e["scan"], "line": e["line"], "level": 1, "title": e["title"], "kind": "matter"}
                        for e in toc if e["scan"] < first or (e["scan"] > last and T.BACK.match(e["title"]))]
        else:
            entries += matter(work, folder, chs[0]["scan"], chs[-1]["scan"])
    order = {"matter": 0, "book": 1, "chapter": 2, "verse": 3}
    entries.sort(key=lambda e: (e["scan"], e["line"], order[e["kind"]]))
    covered = {(c["book"], c["ch"]) for c in chs}
    result = {"work": folder.name, "books": books, "passage": g.get("pas", ""), "toc": entries, "gaps": still,
              "selective": sorted(selective), "chapters": len(chs), "found": {"head": sum(c["how"] in ("head", "top") for c in chs),
                                              "rh": sum(c["how"] == "rh" for c in chs), "margin": sum(c["how"] == "margin" for c in chs),
                                              "text": sum(c["how"] == "text" for c in chs), "scan": sum(c["how"] == "scan" for c in chs),
                                              "rh+scan": sum(c["how"] == "rh+scan" for c in chs)},
              "not_treated": {b: [c for c in range(1, B.CHAPTERS[b] + 1) if (b, c) not in covered] for b in books},
              "cost_usd": round(cost, 5)}
    target = out / f"{work}.json" if out else folder / OUT_NAME
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{work}: {len(chs)} chapters ({result['found']}), {len(entries) - len(chs)} other entries, {len(still)} gaps open, "
          f"${cost:.3f}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*")
    ap.add_argument("--list", type=Path)
    ap.add_argument("--out", type=Path, help="write here, not with the book")
    ap.add_argument("--context", type=Path, help="toc_context results (<work>.json) to take the front and back matter from")
    ap.add_argument("--no-ask", action="store_true", help="no questions to Haiku: the gaps stay open")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--works", dest="list_works", action="store_true", help="print the commentaries that can be done")
    ap.add_argument("--parallel", type=int, default=1)
    args = ap.parse_args()
    if args.list_works:
        for w, g in sorted(works().items()):
            print(w, ";".join(g["bb"]), g.get("pas", ""), sep="\t")
        return
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    todo = [id_of(w) for w in args.works + (args.list.read_text().split() if args.list else [])]

    def one(w):
        try:
            r = run(w, args.out, args.context, not args.no_ask)
            if args.show:
                for e in r["toc"]:
                    print(f"  {e['scan']:>5} {'  ' * (e['level'] - 1)}{e['title'][:90]}  [{e.get('how', e['kind'])}]")
                print("  gaps:", [(g["book"], g["ch"]) for g in r["gaps"]])
        except Exception as e:   # one book's trouble is no reason to stop the others
            print(f"{w}: {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    main()
