"""Words in a transcription that are no Latin words, set right with the dictionary (common/lexicon.py).

Over 39 Latin works (13 November 2026, /data/prdl/testset/work/latin_check/, 1.41 million words) these were the
mechanical faults, beside some 0.25% of real misreadings that no rule finds:
- a word cut at a line end without its hyphen (ignoran / tia): 1.3% of the words; the reader and the translation
  (common/units.read_units) joined the two lines with a space, so "ignoran tia";
- a long s read as f (poteft, noftri, fed): 0.67%;
- a soft hyphen (U+00AD) the units did not rejoin, and HTML entities as text (&shy; &nbsp; &aelig;).

repair_book() mends a book.xml in place, in this order:
1. entities: &shy; becomes a soft hyphen, the others their character (&nbsp; a space);
2. soft hyphens: at the end of a line a hyphen, inside a line nothing (the space after it goes too);
3. long s: in a word the dictionary does not know, an f that is not the last letter becomes s (or ſ, when the book
   writes ſ) if the word is then known; on a page that is Latin (LATIN_SHARE of its words known) only, so an English
   "of" or "fed" stays. A capital F stays (a capital S does not look like it). A word with a tag inside stays;
4. cut words: a body line ending in letters, the next body line beginning in lower case, one of the halves no word
   and the two together a word: the line gets its hyphen, so that read_units joins them. Over a page too: heading
   lines at the top of the next page, before the text goes on, are the running head the model took for a heading
   (moved to <running_head>), unless the page's catchword names the heading rather than the text that goes on.
   A heading inside a page between the halves: left.
The text of a line is otherwise unchanged: what it changes keeps every letter, so a translation still fits its block
(translation/blocks.near).

page_store.Output.merge runs it on every book.xml it makes, so a new transcription and a merge again get it; the
pages (pNNNN.xml) stay as the model answered. For the books made before: transcription/repairs/latin_words.py.
Tested in tests/test_latin_words.py.
"""
import html
import itertools
import re
import xml.etree.ElementTree as ET
from collections import Counter

from common.bookfile import line_markup
from common.lexicon import known, norm

LATIN_SHARE = 0.6         # a page is Latin when this share of its words of 3 letters or more is known
LATIN_MIN_WORDS = 5       # fewer words than this: too few to tell, no long s there
WORD = re.compile(r"[^\W\d_]+")
ENTITY = re.compile(r"&([a-zA-Z]{2,8});")
HYPHENS = "-¬⸗=\xad"
KINDS = ("entity", "soft_hyphen", "long_s", "cut", "cut_page", "running_head")


# ------------------------------------------------------------------ the text nodes of a line

def _nodes(el) -> list:
    """The text nodes of a line in reading order, as [element, "text" | "tail"]."""
    out = [[el, "text"]]
    for c in el:
        out += _nodes(c)
        out.append([c, "tail"])
    return out


def _get(node) -> str:
    return getattr(node[0], node[1]) or ""


def _set(node, value: str):
    setattr(node[0], node[1], value)


def _text(l) -> str:
    return " ".join("".join(l.itertext()).split())


# ------------------------------------------------------------------ 1, 2: entities and soft hyphens

def _entities(l) -> int:
    n = 0

    def one(m):
        nonlocal n
        name = m.group(1)
        if name == "shy":
            n += 1
            return "\xad"
        ch = html.unescape(m.group(0))
        if ch == m.group(0):
            return ch
        n += 1
        return " " if ch == "\xa0" else ch
    for node in _nodes(l):
        t = _get(node)
        if "&" in t:
            _set(node, ENTITY.sub(one, t))
    return n


def _soft_hyphens(l) -> int:
    nodes = [nd for nd in _nodes(l) if _get(nd)]
    if not any("\xad" in _get(nd) for nd in nodes):
        return 0
    n = 0
    last = max(i for i, nd in enumerate(nodes) if _get(nd).strip())
    for i, nd in enumerate(nodes):
        t = _get(nd)
        if "\xad" not in t:
            continue
        if i == last and t.rstrip().endswith("\xad"):
            t = t.rstrip()[:-1] + "-"
            n += 1
        n += t.count("\xad")
        _set(nd, re.sub(r"\xad\s*", "", t))
    return n


# ------------------------------------------------------------------ 3: long s

def long_s_word(word: str, s: str = "s") -> str | None:
    """word with the f that stands for a long s written s (s is "s" or "ſ"), or None: when word is known, has no such
    f, or more than one way makes a word."""
    if known(word):
        return None
    spots = [i for i, ch in enumerate(word[:-1]) if ch == "f"]
    if not spots or len(spots) > 4:
        return None
    for k in range(1, len(spots) + 1):
        hits = []
        for comb in itertools.combinations(spots, k):
            v = "".join(s if i in comb else ch for i, ch in enumerate(word))
            if known(v):
                hits.append(v)
        if hits:
            return hits[0] if len(hits) == 1 else None
    return None


def _is_latin(lines) -> bool:
    words = [w for l in lines for w in WORD.findall(_text(l)) if len(w) >= 3]
    return len(words) >= LATIN_MIN_WORDS and sum(map(known, words)) >= LATIN_SHARE * len(words)


def _long_s_line(l, s: str) -> int:
    """Mend the words with an f for a long s; a word that runs over a tag (pote</i>ft) is left."""
    nodes = _nodes(l)
    texts = [_get(nd) for nd in nodes]
    n = 0
    for k, (nd, t) in enumerate(zip(nodes, texts)):
        if "f" not in t:
            continue
        before = "".join(texts[:k])[-1:]
        after = "".join(texts[k + 1:])[:1]

        def one(m):
            nonlocal n
            w = m.group(0)
            if (m.start() == 0 and before.isalpha()) or (m.end() == len(t) and after.isalpha()):
                return w
            new = long_s_word(w, s)
            if new is None:
                return w
            n += 1
            return new
        _set(nd, WORD.sub(one, t))
    return n


def _uses_long_s(book) -> bool:
    text = "".join(t for l in book.iterfind(".//main_text/l") for t in l.itertext())
    long, short = text.count("ſ"), text.count("s")
    return long > 0 and long >= 0.05 * (long + short)


# ------------------------------------------------------------------ 4: cut words

def _catchword(page) -> str:
    return " ".join((c.text or "") for c in page.iter("catchword")).strip()


def _letters(t: str) -> str:
    return re.sub(r"[^a-z]", "", norm(t))


def _add_hyphen(l):
    nodes = [nd for nd in _nodes(l) if _get(nd).strip()]
    _set(nodes[-1], _get(nodes[-1]).rstrip() + "-")


def _to_running_head(page, heads: list):
    text = " ".join(_text(h) for h in heads)
    for main in page.iterfind("main_text"):
        for h in heads:
            if h in list(main):
                main.remove(h)
    rh = page.find("running_head")
    if rh is None:
        rh = ET.Element("running_head")
        page.insert(0, rh)
    old = (rh.text or "").strip()
    if not old:
        rh.text = text
    elif _letters(text) not in _letters(old):
        rh.text = f"{old} {text}"


def _cuts(book, count: Counter):
    seq = []   # (page, line, heading?) of the text pages in reading order
    for page in book.iter("page"):
        if (page.get("type") or "text") != "text":
            continue
        for l in page.iterfind("main_text/l"):
            if _text(l):
                seq.append((page, l, l.get("type") == "heading"))
    for i, (page, l, head) in enumerate(seq):
        if head:
            continue
        t = _text(l)
        if not t or t[-1] in HYPHENS:
            continue
        m = re.search(r"(?:^|[^\w])([^\W\d_]+)$", t)
        if not m:
            continue
        heads, j = [], i + 1
        while j < len(seq) and seq[j][2]:
            heads.append(seq[j]); j += 1
        if j >= len(seq):
            continue
        page2, l2, _ = seq[j]
        t2 = _text(l2)
        m2 = re.match(r"([^\W\d_]+)", t2)
        if not m2:
            continue
        a, b = m.group(1), m2.group(1)
        if (b[0].isupper() and not a.isupper()) or len(a + b) < 4:
            continue
        if (known(a) and known(b)) or not known(a + b):
            continue
        if heads:
            # only heading lines that open the next page, before its text: a running head read as a heading
            if page2 is page or any(h[0] is not page2 for h in heads):
                continue
            cw = _letters(_catchword(page))
            if cw and not (_letters(t2).startswith(cw) or cw.startswith(_letters(b))):
                continue   # the catchword names the heading: a real heading at the top of the page
            _to_running_head(page2, [h[1] for h in heads])
            count["running_head"] += 1
        _add_hyphen(l)
        count["cut" if page2 is page else "cut_page"] += 1


# ------------------------------------------------------------------ the book

def repair_book(book: ET.Element, changes: list | None = None) -> Counter:
    """Mend book.xml (its root, in place); returns the count of each kind (KINDS). changes, when given, gets every
    line changed: {"scan", "line", "before", "after"} (after None: a heading moved to the running head)."""
    count = Counter({k: 0 for k in KINDS})
    lines = [(page, l) for page in book.iter("page") for l in page.iterfind("main_text/l")]
    before = {id(l): line_markup(l) for _, l in lines}
    for _, l in lines:
        count["entity"] += _entities(l)
        count["soft_hyphen"] += _soft_hyphens(l)
    s = "ſ" if _uses_long_s(book) else "s"
    for page in book.iter("page"):
        page_lines = list(page.iterfind("main_text/l"))
        if page_lines and _is_latin(page_lines):
            for l in page_lines:
                count["long_s"] += _long_s_line(l, s)
    _cuts(book, count)
    if changes is not None:
        alive = {id(l) for page in book.iter("page") for l in page.iterfind("main_text/l")}
        for page, l in lines:
            after = line_markup(l) if id(l) in alive else None
            if after != before[id(l)]:
                changes.append({"scan": int(page.get("n")), "line": l.get("n"), "before": before[id(l)],
                                "after": after})
    return count
