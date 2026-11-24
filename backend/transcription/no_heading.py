"""Headings that are no headings: a whole Bible verse or a whole paragraph that the model marked as a heading
(found 13 November 2026: 15,183 in 1,023 books, /data/prdl/testset/work/headings/REPORT.md).

judge() says of one heading (its lines joined) which kind of non-heading it is, or None for a real heading:
  verse-lemma     "Ad verf. 12. Quotquot autem eum exceperunt ..."   the verse after its marker
  verse-numbered  "9 Oculus videbit eum, & non addet ..."            the verse after its number, in a commentary
  numbered-para   "XLV. Resurgent autem primò omnes homines ..."     a thesis number with its whole paragraph
  long-para       a whole paragraph without a number
  split           a chapter or part with its first verse or its whole paragraph (CAP. IX. Ad verf. 2. Et ecce ...):
                  the heading must stay and the rest go, so it is left as it is (making it text would lose the chapter)
A real heading stays, however long: a chapter, book, part or question with its printed title (CAP. 13. Quod in ...),
and a dedication or title in capitals.

demote_page() makes such a heading on one page a paragraph of its own, its text unchanged:
- the heading's lines lose type and level; its first line starts a paragraph, and so does the line after it;
- a heading as the reader joins it (common/units.read_units: the heading lines at one level in a row on a page) is
  judged by each of the parts the model began (para): all of them no heading, it becomes one paragraph, so that its
  letters stay those of the unit it was; one of them a real heading (THESIS I. with its whole first thesis), it is left.
page_store.py does this to every page it saves, so a new transcription has none; repairs/demote_heads.py did it to the
books transcribed before (with the line after a heading that ends a page, and the translations).
"""

import json
import re
from functools import lru_cache

from common.bookfile import line_markup
from common.paths import genre_answers

TAG = re.compile(r"<[^>]+>")
# a verse marker at the start, with at least one number: Vers. 3 / Verſ. 12 / Ad verf. 4. 5. / v. 21
VERSE_START = re.compile(r"^(?:ad\s+)?(?:vers|verf|verſ|v)[a-zſ]*\.?\s*\d{1,3}\b", re.I)
VERSE_NUMBER = re.compile(r"^\d{1,3}\.?\s+[A-ZÆŒ]")
# a verse marker just after the chapter that opens the heading: CAP. IX. Ad verf. 2. Et ecce ...
VERSE_INSIDE = re.compile(r"(?:^|[\s.])(?:ad\s+)?(?:vers|verf|verſ)[a-zſ]*\.?\s*\d{1,3}\b", re.I)
# a number in front; a lone D, M, C or L is Doctor, Dominus, Magister ..., no number
NUMBER_START = re.compile(r"^(?:§\.?\s*)?(?:(?![DMCL][.,)])[IVXLCDM]{1,8}|\d{1,3})[.,)]\s+\S")
# the printed parts of a book: a heading that opens with one of them is a real heading, however long
STRUCTURE = re.compile(
    r"^(?:cap(?:ut|vt|itulum)?|liber|lib|pars|partis|disputatio|disp|sectio|sect|articulus|art|"
    r"qu(?:æ|ae|e)stio|qu(?:æ|ae)st|locus|loc|tractatus|exercitatio|dissertatio|oratio|epistola|epist|"
    r"pr(?:æ|ae)fatio|thesis|theses|propositio|conclusio|psalmus|psal|homilia|concio|sermo|lectio)"
    r"(?![a-zæœſ])\.?", re.I)
WORD = re.compile(r"[A-Za-zÀ-ÿſæœÆŒ]{2,}")
STOP = re.compile(r"[.;:?!]\s+[A-Za-zÀ-ÿſæœÆŒ]{3,}")   # a full stop, colon or question mark with a word after it

LONG_CHARS, LONG_WORDS = 300, 45   # a whole paragraph: longer than this ...
MID_CHARS, MID_STOPS = 180, 3      # ... or this long with this many stops (with a number in front: one stop)
TITLE_CAPITALS = 0.25              # a dedication or title (AMPLISSIMIS ... VIRIS, D. ...): a paragraph has a few %
# a dedication opens with whom it is for: Amplissimis, Admodum Reverendis, Viris, Domino, To the Right Honourable
DEDICATION = re.compile(r"^(?:admodum|viris?|viro|domin[oi]s?|dn|reverend\w*|pr(?:æ|ae)nobil\w*|to the|"
                        r"[a-zæœ]*(?:issim|ißim|iſſim|isſim)(?:o|is|ae|æ|i|isque|æque)|[a-zæœ]*iss)\b", re.I)
SPLIT_CHARS = 600                  # a chapter heading longer than this has its paragraph in it


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", TAG.sub("", text)).strip()


def capitals(t: str) -> float:
    """The share of capitals among the letters."""
    letters = [c for c in t if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def judge(text: str, commentary: bool = False) -> str | None:
    """The kind of non-heading this heading is (see the module's doc), or None for a real heading.
    commentary: the work is a Bible commentary (haiku_genres.json), where a numbered line is a verse."""
    t = plain(text)
    words = len(WORD.findall(t))
    if capitals(t) >= 0.6:
        return None
    if STRUCTURE.match(t):
        m = re.match(r"cap", t, re.I) and VERSE_INSIDE.search(t, 0, 40)   # a question may name a verse
        if len(t) > SPLIT_CHARS or (m and len(WORD.findall(t[m.end():])) >= 8):
            return "split"
        return None
    if VERSE_START.match(t) and words >= 8:
        return "verse-lemma"
    if commentary and VERSE_NUMBER.match(t) and 8 <= words <= 120:
        return "verse-numbered"
    if capitals(t) >= TITLE_CAPITALS or DEDICATION.match(t):
        return None
    stops = len(STOP.findall(t))
    numbered = bool(NUMBER_START.match(t))
    if len(t) > LONG_CHARS or words > LONG_WORDS or (len(t) > MID_CHARS and stops >= (1 if numbered else MID_STOPS)):
        return "numbered-para" if numbered else "long-para"
    return None



KINDS = {"verse-lemma", "verse-numbered", "numbered-para", "long-para"}   # "split" is never made text


@lru_cache(maxsize=1)
def _genres() -> dict:
    return json.loads(genre_answers().read_text(encoding="utf-8")) if genre_answers().is_file() else {}


def is_commentary(work: str) -> bool:
    """Is the work (W00352) a Bible commentary, as Haiku read it (haiku_genres.json)? There a numbered line is a verse."""
    return (_genres().get(work[:6]) or {}).get("bijbel") == "commentaar"


def _lines(page) -> list:
    """A page's lines as common/units.read_units reads them (a page can have several <main_text>)."""
    return list(page.iterfind("main_text/l"))


def _key(l, i: int) -> str:
    return l.get("n") if (l.get("n") or "").isdigit() else f"#{i}"


def _heading_units(lines: list) -> list[list[int]]:
    """The headings of a page as the reader joins them: the heading lines at one level in a row (an empty line
    between them is skipped, as read_units does). Each as the indexes of its lines."""
    out, cur, level = [], None, None
    for i, l in enumerate(lines):
        if not "".join(l.itertext()).strip():
            continue
        if l.get("type") == "heading":
            lv = l.get("level") or "2"
            if cur is not None and lv == level:
                cur.append(i)
            else:
                cur, level = [i], lv
                out.append(cur)
        else:
            cur = level = None
    return out


def page_plan(page, commentary: bool, kinds=KINDS) -> tuple[dict, list, int, bool]:
    """What to change on one page: ({line key: "first" | "inner" | "para"}, the units made text as
    {"line", "kinds", "starts", "text"}, the units left because a real heading is in them, and whether the
    next page's first line must start a paragraph (a heading made text ended this page)."""
    todo, units, mixed, carry = {}, [], 0, False
    lines = _lines(page)
    for idx in _heading_units(lines):
        starts = [i for i in idx if i == idx[0] or lines[i].get("para") == "true"]
        parts = [" ".join(line_markup(lines[j]) for j in idx if a <= j < b)
                 for a, b in zip(starts, starts[1:] + [idx[-1] + 1])]
        verdicts = [judge(t, commentary) for t in parts]
        if not any(v in kinds for v in verdicts):
            continue
        if not all(v in kinds for v in verdicts):
            mixed += 1
            continue
        for i in idx:
            todo[_key(lines[i], i)] = "first" if i == idx[0] else "inner"
        units.append({"line": int(lines[idx[0]].get("n") or 0), "kinds": verdicts,
                      "starts": [int(lines[i].get("n") or 0) for i in starts], "text": parts})
        after = [j for j in range(idx[-1] + 1, len(lines)) if "".join(lines[j].itertext()).strip()]
        if not after:
            carry = True
        elif lines[after[0]].get("type") != "heading" and lines[after[0]].get("para") != "true":
            todo.setdefault(_key(lines[after[0]], after[0]), "para")
    return todo, units, mixed, carry


def first_line_para(page) -> str | None:
    """The key of a page's first line, when it must start a paragraph after a heading made text on the page before."""
    lines = _lines(page)
    first = [j for j, l in enumerate(lines) if "".join(l.itertext()).strip()]
    if first and lines[first[0]].get("type") != "heading" and lines[first[0]].get("para") != "true":
        return _key(lines[first[0]], first[0])
    return None


def apply_page(page, todo: dict) -> int:
    """Make the changes of plan_book on one page (of book.xml or a pages/pNNNN.xml); returns the lines changed."""
    n = 0
    for i, l in enumerate(_lines(page)):
        what = todo.get(_key(l, i))
        if what in ("first", "inner"):
            l.attrib.pop("type", None)
            l.attrib.pop("level", None)
        if what in ("first", "para"):
            l.set("para", "true")
        elif what == "inner":
            l.attrib.pop("para", None)
        n += bool(what)
    return n


def demote_page(page, commentary: bool = False) -> int:
    """Make the headings on this page that are no headings paragraphs (see the module's doc); returns how many."""
    todo, units, _, _ = page_plan(page, commentary)
    apply_page(page, todo)
    return len(units)
