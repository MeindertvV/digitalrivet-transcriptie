"""The structure of a book: its contents list, chosen once, with where it came from.

This is a stage of its own, after the transcription and before the translation and the site:

    python contents/structure.py W05392           # data/<work>/structure.json
    python contents/structure.py --all            # every book
    python contents/structure.py --report         # how the books stand, per source

A contents list can come from several places; make() takes the first that holds, in this order:
    review    a reviewed list (pilot/<work>/structuur.json; how they were made: archive/structuur_pilot/pilot/PILOT.md)
    context   the outline Haiku made in one request with the text around the headings (toc_context.py),
              unless its own count or doubt() sets it aside
    haiku     the outline Haiku made of the book's headings (toc_haiku.json, by toc_haiku.py, now in archive/),
              unless doubt() sets it aside
    index     the book's own table of contents read from its index pages by Haiku (index_toc.py)
    register  the book's own table of contents in the transcription (register.py)
    heads, heads+headings   the running heads (head_toc), with the headings inside long chapters
    headings  the headings of level 1 in the text
    ""        nothing

structure.json says which one it was and how far to trust it (CONFIDENCE). That is a judgement per
source, not a measurement: backend/eval/ measures it on a set of books kept apart.
The site reads structure.json and has no part in choosing; the file holds the hash of what it was
made from (inputs()), and where that no longer agrees of() makes the list again, so a book without
the file looks the same.
"""

import argparse
import collections
import difflib
import hashlib
import json
import re
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.paths import DATA, refuse_foreign  # noqa: E402
from contents import headings, printed_contents as register, toc_context  # noqa: E402
from contents import index_toc  # noqa: E402
from common.works import id_of, text_of  # noqa: E402
from common.bookfile import EDIT_FILE, load  # noqa: E402
from common.paths import PILOT  # noqa: E402
from common.text import bare_letters as letters, element_text as plain, unless_none as clean  # noqa: E402  (was via enrich/book.py)

OUT_NAME = "structure.json"
# The version of how a book's contents are chosen and made. Raise it in the commit that changes what make() gives
# for a book: every structure.json then no longer holds and is made anew (by the build as it goes, or by
# structure.py --all). A change to the code that leaves the lists as they are leaves this number alone.
STRUCTURE_VERSION = 3   # 2: a letter-spaced label that ran into its letter-spaced number is a bare label (P37); 3: the entries of an earlier part of a printed list stay in that part (P35)
# how far a contents list from each source can be trusted, as a judgement until backend/eval/ has measured it
CONFIDENCE = {"review": "high", "context": "medium", "haiku": "medium", "index": "medium", "register": "medium", "heads+headings": "medium",
              "heads": "medium", "headings": "low", "": "none"}
SPARSE = 100     # scans per entry beyond which a contents list says little of the book
TOC_TITLE_MAX = 110   # longer titles are cut at a word
TOC_KIDS_MIN = 2      # fewer headings inside a chapter of the running heads: no sub-list
TOC_LONG = 40         # a chapter of the running heads this many scans long (and 3x the usual) gets one

# ---------------------------------------------------------------- contents from the running heads

# roman numerals and ordinal words count as the number they spell: LIBER TERTIVS = LIBER TERTIUS = LIBER III
from common.text import ordinal_word as ordinal, roman_word as roman  # noqa: E402,F401  (step 57)
HEAD_MIN = 3        # fewer chapters than this: the heads say nothing, use the headings instead
HEAD_BACK = 2       # head-less pages before a chapter's first head that still belong to it
HEAD_SNAP = 2       # a chapter starts on the page of its own heading when that is this many scans off its first head
HEAD_WINDOW = 4     # a head matching one of this many chapters back is a stray, not a new chapter


class Head:
    """A running head reduced to what survives OCR: its letters (u/v, i/j, æ folded) and its numbers."""
    __slots__ = ("text", "nums", "raw", "uncertain", "stop", "wl")

    def __init__(self, raw: str):
        self.uncertain = "[" in raw   # [?], [Hebrew]: the transcriber was not sure
        s = re.sub(r"\[[^\]]*\]", " ", raw)
        self.stop = s.rstrip().endswith(".")
        self.raw = " ".join(s.split()).strip(" .,;:·")
        s = unicodedata.normalize("NFKD", s.lower().replace("æ", "ae").replace("œ", "oe"))
        words, nums = [], []
        for w in re.findall(r"[a-z]+|\d+", s):   # Hebrew and Greek letters drop out here
            n = int(w) if w.isdigit() else roman(w)
            if n is None:
                n = ordinal(w)
            if n is None:
                words.append(w.replace("v", "u").replace("j", "i"))
            else:
                nums.append(n)
        self.text, self.nums, self.wl = "".join(words), tuple(nums), words

    @property
    def key(self):
        return self.text, self.nums

    def words(self, o) -> bool:
        """The same words, numbers left aside."""
        a, b = sorted((self.text, o.text), key=len)
        if not a:
            return not b
        if a == b or (len(a) >= 4 and a in b):   # DE PECCATO / DE PECCATO ORIGINALI
            return True
        # abbreviated: Loc. XVII / Locus XVII, De Lib. Arbit. In Statu Pec. / De Libero Arbitrio
        s, l = sorted((self.wl, o.wl), key=len)
        # (an abbreviation keeps 3 letters at least: DE is no DEDICATORIA; words more only with the
        # same number: THEOLOGIÆ is not THEOLOG. LIB. SECVND)
        if (len(s) == len(l) or self.nums and self.nums == o.nums) and sum(map(len, s)) >= 3 and all(x == y or min(len(x), len(y)) >= 3 and (x.startswith(y) or y.startswith(x))
                                         for x, y in zip(s, l)):
            return True
        # misread letters, a few at most: Ee Justificatione is De Justificatione, De Sanctificatione is
        # not; a word more or less at either end (EXPLICATIO CATECHIST. / … CATECHET. Pars) is no misreading
        m = difflib.SequenceMatcher(None, a, b)
        diff = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in m.get_opcodes()
                   if op != "equal" and not (op != "replace" and (i1 == j1 == 0 or (i2, j2) == (len(a), len(b)))))
        return m.ratio() >= 0.8 and diff <= max(2, len(b) // 6)

    def same(self, o) -> bool:
        n = min(len(self.nums), len(o.nums))   # CAPVT II is not CAPVT III; CAPVT XIII I is CAPVT XIII
        if self.nums[:n] != o.nums[:n]:
            return False
        if not self.text and not o.text:
            return self.nums == o.nums
        return self.words(o)

    def clash(self, o) -> bool:
        """The same words with other numbers: LOCUS II against LOCUS I."""
        n = min(len(self.nums), len(o.nums))
        return n > 0 and self.nums[:n] != o.nums[:n] and self.words(o)

    def again(self, o) -> bool:
        """Is o this head printed again, its number too? LOCUS without one does not repeat LOCUS VIII."""
        return self.same(o) and o.nums[:len(self.nums)] == self.nums


def unnumber(heads):
    """Drop the numbers of heads whose numbers are noise: per family of heads with the same words,
    the numbers in the order of the pages should go on and not keep falling back to one just left
    (Pars 2, Pars 1, Pars 2, Part 3, Pars 2 in a whole part 2: half the changes or more). A
    misprint now and then (LOCUS VIII amid LOCUS VII) is no reason, nor starting again at I in a
    new book."""
    families = []   # [first head, [heads with a number]]
    for h in heads:
        f = next((f for f in families if f[0].words(h)), None)
        if f is None:
            families.append(f := [h, []])
        if h.nums:
            f[1].append(h)
    for _, hs in families:
        changes = back = 0
        recent = []   # the numbers before the current one, last first
        for a, b in zip(hs, hs[1:]):
            if a.nums[0] != b.nums[0]:
                changes += 1
                back += b.nums[0] in recent[:3]
                recent = [a.nums[0]] + [n for n in recent if n != a.nums[0]]
        if back >= 3 and back * 2 >= changes:   # W07389: 9 in 10 changes go back, W03723 1 in 3
            for h in hs:
                h.nums = ()


def belongs(h, c) -> bool:
    """Does head h belong to chapter c? Not when the chapter has these words with another number
    (LOCUS II in the chapter of LOCUS I), even if it also has them without one (LOCUS, its I unread);
    a number printed on one page only (Pars 1, misread amid Pars 3) does not count."""
    return any(h.same(o) for o in c.heads) and not any(h.clash(o) for o in c.firm())


def head_toc(heads, printed=None):
    """Contents list from the running heads: [(scan index, title)].

    heads: (scan index, running head) for every page in order, "" where there is none.
    printed: {scan index: printed page number}, to tell left pages from right ones.
    Rules, from how the books print their heads:
    - A chapter is a run of pages whose heads match each other, OCR noise allowed (Head.same).
      Left and right pages often differ (LIBRI PRIMI / CAPVT II), so a head also matches when it
      is one of the previous chapter's: the other side of the spread can carry on unchanged.
    - A head that matches none of these starts a chapter; one matching a chapter a little further
      back, marked uncertain ([?]), or not printed again on its side of the next spread while that
      side returns to the current chapter there or on the spread after (a misprinted LOCUS VIII
      amid LOCUS VII, a table's head, Pars 1 misread amid Pars 2) is a stray and stays in the
      current chapter.
    - A title can be set over the spread (DE SPIRITV / SANCTO.): the next new head is the rest of
      the title when the chapter has one page so far and either that head has no closing full stop
      while the new one has, or the head after the new one repeats the first. Two chapters of one
      page on consecutive scans (DE MERITIS. / HOMINVM.) are one title too.
    - A chapter opens on a page without a head (its title is set large), so it starts on the
      head-less pages right before its first head (at most HEAD_BACK).
    - A title set over the spread reads from the left page to the right one (Chapter.title); a
      part the book puts on the left page in several such titles (LOCUS) comes first in the
      chapters too short to show their sides."""
    def side(i):
        m = re.search(r"\d+", (printed or {}).get(i, ""))
        return 0 if not m else 1 if int(m.group()) % 2 == 0 else -1

    seq = [(i, Head(h)) for i, h in heads if h]
    seq = [(i, h) for i, h in seq if h.text or h.nums]
    unnumber(h for _, h in seq)
    at = dict(seq)
    chapters = []
    for k, (i, h) in enumerate(seq):
        if chapters:
            cur = chapters[-1]
            if any(belongs(h, c) for c in chapters[-2:]):
                cur.add(i, h, side(i))
                continue
            if h.uncertain or any(belongs(h, c) for c in chapters[-2 - HEAD_WINDOW:-2]):
                continue
            if len(cur.pages) == 1 and len(cur.parts) == 1:
                first = cur.heads[0]
                after = seq[k + 1][1] if k + 1 < len(seq) else None
                if (not first.stop and h.stop) or (after is not None and after.same(first)):
                    cur.add(i, h, side(i), part=True)
                    continue
            side_on = [at[j] for j in (i + 2, i + 4) if j in at]   # the same side of the next two spreads
            if side_on and not h.again(side_on[0]) and any(belongs(o, cur) for o in side_on):
                continue
        chapters.append(Chapter(i, h, side(i)))

    merged = []
    for c in chapters:
        last = merged[-1] if merged else None
        if (last and len(last.pages) == 1 and len(last.parts) == 1 and len(c.pages) == 1
                and c.pages[0] == last.pages[0] + 1):
            last.add(c.pages[0], c.heads[0], side(c.pages[0]), part=True)
        elif (last and len(last.parts) == 1 and len(c.parts) == 1
              and sum(last.heads[0].again(o) for o in last.on) <= 3
              and any(last.heads[0].again(o) for o in c.on[1:])):
            # last is the other side of c's spreads, printed ahead of it: its head returns in c
            # (Loc. VII. De Peccato In Gen. Et In Spec. before In Genere Et In Specie. / Locus VII.
            # De Peccato; pages bound out of order: Tum Eccl. Tum Civil. before Loc XV.)
            f, part = last.heads[0], [last.heads[0], dict(last.parts[0][1]), last.parts[0][2]]
            for i, o in zip(c.pages[1:], c.on[1:]):
                if f.same(o) and not c.heads[0].same(o):
                    part[1][o.raw] = part[1].get(o.raw, 0) + 1
                    part[2] += side(i)
            c.parts.append(part)
            if all(any(o.same(x) for x in c.heads + [f]) for o in last.on):   # last is all this chapter's
                c.pages = last.pages + c.pages
                c.on = last.on + c.on
            merged[-1] = c
        else:
            merged.append(c)

    lefts = {}   # words of the left part of each title whose sides are clear
    for c in merged:
        if c.sided():
            t = c.title_parts()[0][0].text
            lefts[t] = lefts.get(t, 0) + 1
    left = {t for t, n in lefts.items() if n >= 2}

    out, idxs = [], [i for i, _ in heads]
    has = {i for i, h in heads if h}
    for c in merged:
        p = idxs.index(c.pages[0])
        for _ in range(HEAD_BACK):   # stops at the previous chapter's last head
            if p == 0 or idxs[p - 1] in has:
                break
            p -= 1
        out.append((idxs[p], c.title(left)))
    return out


class Chapter:
    """Pages of one chapter in head_toc: their heads, and per part of the title the variants
    printed of the head that founded it and on which side of the spread it is printed (side:
    + for each page with an even printed number, a left page, - for each odd one)."""

    def __init__(self, i, h, side=0):
        self.pages, self.heads, self.parts = [i], [h], [[h, {h.raw: 1}, side]]
        self.on = [h]             # the head of each page
        self.count = {h.key: 1}   # pages per head

    def firm(self):
        """The heads that settle the chapter's numbers: its first one, and those on several pages."""
        return [o for o in self.heads if o is self.heads[0] or self.count[o.key] >= 2]

    def add(self, i, h, side=0, part=False):
        self.pages.append(i)
        self.on.append(h)
        self.count[h.key] = self.count.get(h.key, 0) + 1
        # a head of the previous chapter's (EXPLICATIO Pars I on a left page) is no key to this one
        # when it contradicts its number (Pars 2)
        if all(h.key != o.key for o in self.heads) and not any(h.clash(o) for o in self.firm()):
            self.heads.append(h)
        if part:
            self.parts.append([h, {h.raw: 1}, side])
            return
        for p in self.parts:
            if h.key == p[0].key:
                p[1][h.raw] = p[1].get(h.raw, 0) + 1
                p[2] += side

    def sided(self) -> bool:
        """Two parts, each kept to its own side of the spread on several pages."""
        if len(self.parts) != 2:
            return False
        a, b = sorted(p[2] for p in self.parts)
        return a <= -2 and b >= 2

    def title_parts(self, left=()):
        # a title set over the spread reads from the left page to the right one
        # (ADVERSVS M. FL. ILLYR. DEMONST. / ET EIVSDEM APOLOGIAM.), whichever came first in the book;
        # without clear sides a part the book puts on the left (left: their words) goes first, else
        # the pages' order holds: a title of one page each (DE HOMINE / VETERI.) reads as printed
        parts = self.parts
        if self.sided():
            return sorted(parts, key=lambda p: -p[2])
        if len(parts) == 2 and parts[1][0].text in left and parts[0][0].text not in left:
            return parts[::-1]
        return parts

    def title(self, left=()):
        # per part the variant printed most often, the first one on a tie
        return " ".join(max(v, key=lambda r: (v[r], -list(v).index(r))) for _, v, _ in self.title_parts(left))


# ---------------------------------------------------------------- the contents list

# words that only number a division: "LOCVS II." or "CAP. I." says nothing without the title below it
LABEL = re.compile(r"(?:locus|loc|caput|cap|capitulum|liber|lib|pars|part|disputatio|disp|dissertatio|exercitatio|"
                   r"sectio|sect|quaestio|quaest|q|tabula|tab|articulus|art|thesis|theses|classis|ordo|tractatus|"
                   r"membrum|titulus|tit|et|ac|prim\w*|secund\w*|terti\w*|quart\w*|quint\w*|sext\w*|septim\w*|"
                   r"octau\w*|non\w*|decim\w*|undecim\w*|duodecim\w*|uigesim\w*|uicesim\w*|tri[cg]esim\w*|ultim\w*)")
# closing words the transcription sometimes tags as a chapter heading (as chapters.py)
CLOSING = re.compile(r"^(?:DIXI|FINIS|AMEN|TELOS|LAUS DEO|SOLI DEO GLORIA)\b", re.I)


def bare(title: str) -> bool:
    """Only a label and a number: LOCVS II., CAP. I., C A P. VII., LOCVS DECIMVS."""
    spaced = unicodedata.normalize("NFKD", title.lower())
    s = re.sub(r"\b([a-z])\.? (?=[a-z]\b)", r"\1", spaced)   # letter-spaced: C. A P. X X V.
    words = re.findall(r"[a-z]+|\d+", s)
    joined = set(words) - set(re.findall(r"[a-z]+|\d+", spaced))

    def label(w: str) -> bool:
        return bool(LABEL.fullmatch(w.replace("v", "u").replace("j", "i")))

    def ran_together(w: str) -> bool:   # a spaced label and its spaced number became one word: capxxv
        return w in joined and any(label(w[:i]) and roman(w[i:]) is not None for i in range(2, len(w)))
    return all(w.isdigit() or roman(w) is not None or label(w) or ran_together(w) for w in words)


def numeral(title: str) -> bool:
    """Only a number: VII., 12."""
    words = re.findall(r"[a-z]+|\d+", title.lower())
    return all(w.isdigit() or roman(w) is not None for w in words)


def cut_title(title: str, most: int = TOC_TITLE_MAX) -> str:
    title = title.strip(" ,;:|")
    if len(title) <= most:
        return title
    return title[:most].rsplit(" ", 1)[0].rstrip(" ,;:") + " …"


def join_lines(parts: list[str]) -> str:
    """Lines of one heading as one title: a word broken at the line end is joined again."""
    out = ""
    for p in parts:
        out = out[:-1] + p if out.endswith("-") else f"{out} {p}".strip()
    return out


def heading_entries(root) -> list[tuple[int, int, str]]:
    """The headings of level 1 in the text: [(scan index, line index on the page, title)].

    A heading printed over several lines is one: its first line has para="true" (ocr_split.py,
    mark_heads.py), or in books transcribed without those marks, its lines follow each other on one
    page. A heading that is only a label and a number (LOCVS II.) gets the heading below it as its
    title. A heading that is a number only (VII., a thesis set apart) is no chapter, and the same
    heading again at the top of the next page (a chapter heading repeated over a page break) is
    left out."""
    out = []
    for pg in root.iter("page"):
        idx = int(pg.get("n"))
        main = pg.find("main_text")   # the lines render_pages shows
        ls = [l for l in main if l.tag == "l"] if main is not None else []
        marked = any(l.get("type") == "heading" and l.get("para") == "true" for l in ls)

        def group(k, level=None):
            """The lines of the heading starting at line k: (their texts, the line after them)."""
            lvl = level or ls[k].get("level") or "2"
            parts, j = [plain(ls[k])], k + 1
            while (j < len(ls) and ls[j].get("type") == "heading" and (ls[j].get("level") or "2") == lvl
                   and not (marked and ls[j].get("para") == "true")):
                parts.append(plain(ls[j]))
                j += 1
            return parts, j

        k = 0
        while k < len(ls):
            l = ls[k]
            if l.get("type") != "heading" or (l.get("level") or "2") != "1" or CLOSING.match(plain(l)):
                k += 1
                continue
            parts, j = group(k)
            title = join_lines(parts)
            if bare(title) and j < len(ls) and ls[j].get("type") == "heading":
                more, j = group(j)
                title = f"{title} {join_lines(more)}"
            prev = out[-1] if out else None
            if numeral(title):   # a number alone (a thesis number set as a heading): no chapter
                k = j
                continue
            if not (prev and idx - prev[0] <= 1 and letters(prev[2]) == letters(title)):
                out.append((idx, k, cut_title(title)))
            k = j
    return out


def heading_texts(root) -> list[tuple[int, int, str]]:
    """Every heading line of the text, of any level, and each pair of neighbouring ones read together:
    [(scan index, line index, text)]. A chapter's title is often only part of what is printed over it
    (RERVM FRISICARVM HISTORIAE as level 1, LIBER SEPTIMVS. as level 2 below it)."""
    out = []
    for pg in root.iter("page"):
        main = pg.find("main_text")
        ls = [l for l in main if l.tag == "l"] if main is not None else []
        hs = [(k, plain(l)) for k, l in enumerate(ls) if l.get("type") == "heading"]
        out += [(int(pg.get("n")), k, t) for k, t in hs]
        out += [(int(pg.get("n")), a[0], f"{a[1]} {b[1]}") for a, b in zip(hs, hs[1:])]
    return out


def snap_to_headings(by_head: dict, headings) -> dict:
    """{scan index: title} with each chapter moved to the page of its own heading, when one with the
    chapter's words is printed up to HEAD_SNAP scans before or after its first running head: the
    first head of a chapter is often on the page after the one that opens it (the big LIBER OCTAVVS.
    on a page whose head is still the book's title), and now and then the other way round. The nearest
    such heading wins, and not beyond the neighbouring chapters."""
    starts = sorted(by_head)
    out = {}
    for n, s in enumerate(starts):
        lo = starts[n - 1] + 1 if n else float("-inf")
        hi = starts[n + 1] - 1 if n + 1 < len(starts) else float("inf")
        h = Head(by_head[s])
        near = [i for i, _, t in headings if abs(i - s) <= HEAD_SNAP and lo <= i <= hi and h.same(Head(t))]
        out.setdefault(min(near, key=lambda i: (abs(i - s), i)) if near and s not in near else s, by_head[s])
    return out


def toc_plan(root, edited: dict | None = None) -> tuple[list[tuple[int, int | None, int, str]], str]:
    """The contents list: [(scan index, line index on the page or None for its top, level, title)],
    and where it comes from ("context", "haiku", "index", "register", "heads", "heads+headings", "headings", "review", or "" for none).

    From the first source that holds:
    a. the outline Haiku made of the book's headings (toc_context.py, else toc_haiku.py), the marks in
       the text as it says;
    0. the book's own table of contents read from its index pages by Haiku (index_toc.py), levels as
       printed, also for a book that is only the PDF's OCR;
    1. the book's own table of contents (register.py), levels as printed;
    2. the running heads (head_toc), when they give HEAD_MIN chapters or more (a head that is only
       a label and a number, CAPVT III, gets the title of the heading on the chapter's first page); a long chapter
       (TOC_LONG scans, 3 times the usual) with TOC_KIDS_MIN headings of level 1 or more inside it
       (LOCVS II., III., … under the running head LOCI COMMVNES) gets them as a second level;
    3. the headings of level 1 in the text.
    A book that is only the PDF's OCR has unread heads too, too noisy for chapters: only 3.
    reviewed_plan passes the proposal's titles in edited ({(scan index, line index): title} of the
    headings the reader changed) before the titles made from the line are set aside: one of those
    is a title of its own now."""
    outline, source = haiku_outline(root, edited)
    if outline:
        at = {(int(pg.get("n")), l.get("n")): k for pg in root.iter("page") for k, l in enumerate(headings.lines_of(pg))}
        return [(e["scan"], at.get((e["scan"], str(e["line"]))), e["level"], e["title"]) for e in outline], source
    return plain_plan(root, edited)


OUTLINES = {}   # per book being built: the outline that passed, so the marks it set do not change the verdict


def haiku_outline(root, edited: dict | None = None) -> tuple[list[dict], str]:
    """The first outline by Haiku that headings.doubt does not find worse than what the book had, and
    its source. With none, the book keeps the contents made without it, and its marks in the text."""
    if id(root) not in OUTLINES:
        OUTLINES[id(root)] = [], ""
        for source, read in (("context", toc_context.toc), ("haiku", headings.toc)):
            outline = read(root)
            if outline:
                plan, before = plain_plan(root, edited)
                why = headings.doubt(outline, root, len(plan), before)
                if not why:
                    OUTLINES[id(root)] = outline, source
                    break
                print(f"  contents by {source} set aside: {why}")
    return OUTLINES[id(root)]


def plain_plan(root, edited: dict | None = None) -> tuple[list[tuple[int, int | None, int, str]], str]:
    """The contents list without Haiku's outline: toc_plan from step 0 on."""
    entries, source = next(plain_sources(root, edited), (None, ""))
    return (entries, source) if entries is not None else ([], "")


def sources(root, work: str | None = None) -> list[tuple[str, list[tuple[int, int | None, int, str]]]]:
    """Every contents list the book has, one per source, in the order make() tries them: the reviewed list, the
    outlines of Haiku (also one that doubt() set aside), then those of plain_plan. For the reader, which shows the
    chosen one and lets the reader pick another (build_reader.toc_alternatives). Call it before the outline's marks
    are set in the text (headings.apply), so that the headings are the transcription's own."""
    out = []
    reviewed = reviewed_plan(root, work)
    if reviewed:
        out.append(("review", reviewed[0]))
    at = {(int(pg.get("n")), l.get("n")): k for pg in root.iter("page") for k, l in enumerate(headings.lines_of(pg))}
    for source, read in (("context", toc_context.toc), ("haiku", headings.toc)):
        outline = read(root)
        if outline:
            out.append((source, [(e["scan"], at.get((e["scan"], str(e["line"]))), e["level"], e["title"]) for e in outline]))
    out += [(source, entries) for entries, source in plain_sources(root)]
    return out


def plain_sources(root, edited: dict | None = None):
    """The contents lists of plain_plan, one by one in its order, as (entries, source): only as far as asked."""
    read = index_toc.toc(root)
    if read:
        yield [(i, None, level, title) for level, i, title in read], "index"
    ocr = root.get("kind") == "ocr"
    if not ocr:
        own = register.register(root)
        if own:
            yield [(i, None, level, title) for level, i, title in own], "register"
    headings = heading_entries(root)
    pages = list(root.iter("page"))
    by_head = {} if ocr else dict(
        head_toc([(int(pg.get("n")), clean(pg.findtext("running_head"))) for pg in pages],
                 {int(pg.get("n")): clean(pg.findtext("page_number")) for pg in pages}))
    edited = edited or {}   # {(scan index, line index): title} the reader gave a heading
    if len(by_head) >= HEAD_MIN:
        by_head = snap_to_headings(by_head, heading_texts(root))
    if len(by_head) < HEAD_MIN:
        if headings:
            yield [(i, k, 1, edited.get((i, k)) or t) for i, k, t in headings], "headings"
        return
    starts = sorted(by_head)
    ends = starts[1:] + [int(pages[-1].get("n")) + 1]
    sizes = sorted(e - s for s, e in zip(starts, ends))
    long = max(TOC_LONG, 3 * sizes[len(sizes) // 2])
    out = []
    for s, e in zip(starts, ends):
        title = by_head[s]
        if bare(title):   # CAPVT III: the heading on the chapter's first pages has its title too
            h = Head(title)
            title = next((t for i, _, t in headings if s <= i <= s + HEAD_BACK and not bare(t) and h.same(Head(t))),
                         title)
        out.append((s, None, 1, title))
        if e - s < long:   # the heads divide the book finely enough; headings inside are its sections
            continue
        # the chapter's own heading is no sub-chapter
        kids = [(i, k, 2, edited.get((i, k)) or t) for i, k, t in headings
                if s <= i < e and not (Head(t).same(Head(by_head[s])) and (i, k) not in edited)]
        # nor is a bare number that comes back again and again (TABVLA II. of every topic)
        seen = collections.Counter(letters(t) for *_, t in kids if bare(t))
        kids = [x for x in kids if not (bare(x[3]) and seen[letters(x[3])] > 1)]
        if len(kids) >= TOC_KIDS_MIN:
            out += kids
    yield out, "heads+headings" if any(e[2] > 1 for e in out) else "heads"
    if headings:   # after the running heads, the headings alone are one more list to choose
        yield [(i, k, 1, edited.get((i, k)) or t) for i, k, t in headings], "headings"


def reviewed_plan(root, work: str | None):
    """The contents list from pilot/<work>/structuur.json, if the book was reviewed.

    The reviewed entries carry the scan and printed line of their start. The reader's anchors use
    the line's index on the page, so the printed line is looked up here. An entry whose line is
    gone (a transcription change) is left out rather than put on the wrong line; an entry at the
    top of a page (line 1) keeps its page anchor."""
    if not work:
        return None
    path = PILOT / work / "structuur.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pages = {int(pg.get("n")): pg for pg in root.iter("page")}
    # a heading the reader changed or added gets its own title (the line's says what the reader wanted)
    edited = {(int(e["start"]["scan"]), int(e["start"]["line"])): " ".join(str(e["title"]).split())
              for e in data.get("entries") or [] if e.get("source") == "reader"}
    out, dropped = [], 0
    for e in data.get("entries") or []:
        try:
            scan = int(e["start"]["scan"])
            line = int(e["start"]["line"])
            level = int(e["level"])
        except (KeyError, TypeError, ValueError):
            continue
        pg = pages.get(scan)
        if pg is None:
            dropped += 1
            continue
        main = pg.find("main_text")
        ls = [l for l in (main if main is not None else []) if l.tag == "l"]
        k = None
        for i, l in enumerate(ls):
            try:
                if int(l.get("n")) == line:
                    k = i
                    break
            except (TypeError, ValueError):
                continue
        if k is None and line != 1:
            dropped += 1
            continue
        title = " ".join(str(e.get("display") or e["title"]).split())
        out.append((scan, k, level, title))
    if not out:
        return None
    mode = data.get("page_display") or "printed"
    if dropped:
        print(f"  reviewed contents: {len(out)} entries, {dropped} start(s) no longer in the text")
    return out, "review", mode


def make(root, work: str | None) -> dict:
    """The structure of a book: source, confidence, page_mode ("printed" or "scan"), entries
    [[scan, line index on the page or None for its top, level, title]], and outline, the marks that
    Haiku's outline sets in the text (headings.apply)."""
    reviewed = reviewed_plan(root, work)
    outline = []
    if reviewed:
        entries, source, page_mode = reviewed
    else:
        outline, _ = haiku_outline(root)
        headings.apply(root, outline)
        entries, source = toc_plan(root)
        OUTLINES.pop(id(root))   # a later book can get the same id
        page_mode = "printed"
    scans = sum(1 for _ in root.iter("page"))
    confidence = CONFIDENCE[source]
    if confidence == "medium" and len(entries) * SPARSE < scans:
        confidence = "low"
    return {"source": source, "confidence": confidence, "page_mode": page_mode,
            "entries": [list(e) for e in entries], "outline": outline}


def inputs(book_dir: Path, work: str) -> str:
    """Hash of everything make() reads: the book, the answers of the models and the reviewed list, with the
    version of the code that chooses (STRUCTURE_VERSION)."""
    folder = book_dir.parent if book_dir.name == "ocr_layer" else book_dir
    h = hashlib.sha1()
    for f in [book_dir / "book.xml", folder / toc_context.OUT_NAME, folder / headings.OUT_NAME, folder / index_toc.OUT_NAME,
              PILOT / work / "structuur.json", PILOT / work / EDIT_FILE]:
        h.update(f.name.encode() + b"\0" + (hashlib.sha1(f.read_bytes()).digest() if f.is_file() else b""))
    h.update(f"version {STRUCTURE_VERSION}".encode())
    return h.hexdigest()


def of(root, book_dir: Path, work: str) -> dict:
    """The structure of the book in root, marks in the text set: structure.json when it still holds
    for this book, else made here."""
    path = book_dir / OUT_NAME
    if path.is_file():
        s = json.loads(path.read_text(encoding="utf-8"))
        if s.get("inputs") == inputs(book_dir, work):
            if s["source"] != "review":
                headings.apply(root, s["outline"])
            return s
    return make(root, work)


def write(book_dir: Path, work: str) -> dict:
    """Make data/<work>/structure.json."""
    root, *_ = load(book_dir, work)
    s = {"inputs": inputs(book_dir, work), **make(root, work)}
    path = book_dir / OUT_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)
    return s


def books() -> dict[str, Path]:
    """{work id: folder of its text}: the transcription, else the PDF's OCR."""
    out = {}
    for d in sorted(DATA.iterdir()):
        xml = text_of(d)
        if xml.is_file():
            out[d.name.split("_")[0]] = xml.parent
    return out


def one(item) -> tuple[str, str, str]:
    work, book = item
    s = write(book, work)
    return work, s["source"], s["confidence"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="work ids")
    ap.add_argument("--all", action="store_true", help="every book that has none that still holds")
    ap.add_argument("--report", action="store_true", help="how the books stand, nothing written")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if not args.report:
        refuse_foreign(DATA)
    found = books()
    args.works = [id_of(w) for w in args.works]
    if args.report:
        count, stale = collections.Counter(), 0
        for work, book in found.items():
            path = book / OUT_NAME
            if not path.is_file():
                count[("(no structure.json)", "")] += 1
                continue
            s = json.loads(path.read_text(encoding="utf-8"))
            count[(s["source"] or "(none)", s["confidence"])] += 1
            stale += s.get("inputs") != inputs(book, work)
        for (source, confidence), n in count.most_common():
            print(f"{n:6d}  {source:<20} {confidence}")
        print(f"{stale} no longer agree with what they were made from")
        return
    if not (args.works or args.all):
        ap.error("name works, or --all or --report")
    todo = [w for w in (args.works or found) if args.works or not fresh(found[w], w)]
    if not todo:
        print("every structure.json still holds: nothing to do")
        return
    with ProcessPoolExecutor(args.workers) as pool:
        for k, (work, source, confidence) in enumerate(pool.map(one, [(w, found[w]) for w in todo]), 1):
            print(f"{k}/{len(todo)} {work}: {source or '(none)'}, {confidence}")


def fresh(book_dir: Path, work: str) -> bool:
    path = book_dir / OUT_NAME
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("inputs") == inputs(book_dir, work)


if __name__ == "__main__":
    main()
