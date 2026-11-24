"""Restore the printed lines of a page transcribed paragraph by paragraph, from the PDF's OCR layer.

The model transcribes the running text as paragraphs (prompts/early_modern_latin_paragraphs.txt):
it no longer has to keep track of line breaks. The OCR layer of the scan does know where every
printed line ends, but reads the letters badly. So the letters of both are aligned (long s/f, u/v,
i/j and ligatures folded away, as in enrich/pdfboxes.py), and every OCR line end that falls in a
well-aligned stretch becomes a line break in the transcription, at the nearest space (or after the
hyphen of a broken word, "ever-so" -> "ever-" / "so").

split_page() turns the model's <page> with <head>/<p> blocks into the usual <page> with numbered
<l> lines and <marginalia>, so book.xml, book.json, the checks and the reader stay as they were.
The first line of every paragraph and of every heading gets para="true" (mark_heads() adds it to
the headings of pages split before that).

Test on a transcribed page without the API:
  python transcription/split_lines.py data/<book>/pages/p0012.para.xml E:/PRDL_pdfs/<book>.pdf 12
"""

import copy
import re
import sys
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import lineboxes  # noqa: E402  (ocr_lines, norm)

MIN_OCR_WORDS = 15   # fewer words of 3+ letters on a scan: no usable text layer (as ocr_regels.py)
MIN_WORDLIKE = 0.4  # share of OCR words that must look like words (3+ letters, nothing else); text pages score 0.5-0.9
MIN_BLOCK = 3       # aligned stretches shorter than this are chance
LINE_FOUND = 0.3     # share of an OCR line's letters that must be aligned for its end to count
WEAK = 0.5           # below this share of aligned letters the page gets a warning


HYPHENS = ("-", "¬", "=", "⸗", "�")   # a line-end hyphen in the OCR (often the replacement character)
# a gathering signature ("A 3", "Biij", "Aa 2", "b ij", ")( iiij", "¶ 2."): a letter (perhaps doubled) or a sign,
# then the leaf: one digit, or a Roman number of the few leaves that are signed (signature_line)
SIGNATURE = re.compile(r"(?P<mark>(?P<l>[A-Za-z])(?i:(?P=l)){0,2}|\)\s?\(|\*{1,3}|[†‡¶]{1,2})(?P<sp>\s?)(?P<leaf>\d|[ivxj]{1,5}|[IVX]{1,4})\.?")
LEAVES = {"ij", "ii", "iij", "iii", "iiij", "iiii", "iv", "v", "vj", "vi", "vij", "vii"}
DOTLESS = str.maketrans("ıȷ", "ij")


from common.text import pairing_letters_and_digits as norm  # noqa: E402,F401  (step 57)


def printed_lines(page) -> list[tuple[str, bool]] | None:
    """The printed lines of a pymupdf page from its OCR layer, in reading order: (normalised
    letters, ends with a hyphen), margin words left out. Pieces of one printed line that the OCR
    keeps apart are joined, left to right. Two pieces of a column can be on one line when they
    overlap in height for more than half the lower one, and side to side for less than 1.5 line
    heights (OCR boxes are loose; lines below each other overlap over most of their width). Such
    pairs are joined best first, by how much they overlap in height for the taller one, whatever
    order the OCR has them in (a heading number that starts higher goes with the rest of its
    heading, not with the line above; a large initial or a heading number that overlaps a line's
    piece over its whole height does not take it from the rest of its line), and two lines only become
    one when none of their pieces overlap side to side: a large initial or a speck between two
    lines does not glue them together, a piece in a gap of a line (a stray comma) still joins it.
    ocr_lines sorts each column by height, so a column starts where the order goes back up the
    page. None when the page has no usable text layer."""
    ocr = lineboxes.ocr_lines(page)
    if sum(len(lineboxes.norm(w)) >= 3 for o in ocr for w in o[1].split()) < MIN_OCR_WORDS:
        return None
    words = [w.strip(".,;:?!()'\"-") for o in ocr for w in o[0].split()]
    if sum(len(w) >= 3 and w.isalpha() for w in words) < MIN_WORDLIKE * len(words):
        return None   # noise, e.g. a page the OCR read upside down
    hs = sorted(o[2][3] - o[2][1] for o in ocr)
    slack = 1.5 * hs[len(hs) // 2]  # side-to-side overlap pieces of one line may have
    pieces, col, mid = [], 0, None   # (x0, text, x1, y0, y1, column)
    for _, inner, (x0, y0, x1, y1) in ocr:
        if mid is not None and (y0 + y1) / 2 < mid - (y1 - y0):
            col += 1
        mid = (y0 + y1) / 2
        pieces.append((x0, inner, x1, y0, y1, col))

    def apart(a, b):
        return min(a[2], b[2]) - max(a[0], b[0]) < slack

    pairs = []
    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            b = pieces[j]
            if b[5] != a[5]:
                break
            both = min(a[4], b[4]) - max(a[3], b[3])
            v = both / max(min(a[4] - a[3], b[4] - b[3]), 1)
            if v > 0.5 and apart(a, b):   # pieces of one line are alike in height, unlike an initial
                pairs.append((-both / max(a[4] - a[3], b[4] - b[3], 1), i, j))
    line = list(range(len(pieces)))   # piece -> the line it is on, named by one of its pieces
    members = {i: [i] for i in line}
    for _, i, j in sorted(pairs):
        a, b = line[i], line[j]
        if a != b and all(apart(pieces[p], pieces[q]) for p in members[a] for q in members[b]):
            for q in members[b]:
                line[q] = a
            members[a] += members.pop(b)
    lines = []
    for m in sorted(members.values(), key=min):
        parts = sorted(pieces[q] for q in m)
        lines.append((norm(" ".join(q[1] for q in parts)), parts[-1][1].rstrip().endswith(HYPHENS)))
    return [l for l in lines if l[0]]


ITALIC, BLACK = 1, 2   # the type of a character (flatten): italics, black letter


def flatten(el):
    """A block as a list of (character, type) with whitespace collapsed, and its marginal notes
    as (position in that list, <note>). type: ITALIC and BLACK added up (0: roman)."""
    chars, notes = [], []

    def add(t, it):
        for c in t or "":
            if c.isspace():
                if chars and chars[-1][0] != " ":
                    chars.append((" ", it))
            else:
                chars.append((c, it))

    def walk(e, it):
        add(e.text, it)
        for c in e:
            if c.tag == "note":
                n = copy.deepcopy(c)
                n.tail = None
                notes.append((len(chars), n))
            else:
                walk(c, it | ITALIC * (c.tag in ("i", "hi")) | BLACK * (c.tag == "bl"))
            add(c.tail, it)

    walk(el, 0)
    while chars and chars[-1][0] == " ":
        chars.pop()
    return chars, notes


SENTENCE_END = ".?!"


def sentence_start(chars, pos: int) -> int:
    """The start of the sentence that position pos is in: just after the last ". ", "? " or "! "
    before it, or the start of the block."""
    k = min(pos, len(chars))
    while k > 0:
        if chars[k - 1][0] == " " and k >= 2 and chars[k - 2][0] in SENTENCE_END:
            return k
        k -= 1
    return 0


def anchor_notes(blocks) -> None:
    """Every marginal note to the start of the sentence it stands beside. A note with at="first
    words" (prompts/layout/_common.txt, rule A) goes where those words start, in its own block or
    else the nearest block that has them; a note without goes back from where the model put it
    (often in mid-sentence) to the start of that sentence. Changes blocks in place."""
    texts = []
    for chars, _ in blocks:
        t, at = [], []
        for c, (ch, _) in enumerate(chars):
            for l in norm(ch):
                t.append(l)
                at.append(c)
        texts.append(("".join(t), at))
    moves = []
    for b, (chars, notes) in enumerate(blocks):
        for i, (pos, note) in enumerate(notes):
            key = norm(note.get("at") or "")
            dest = None
            if len(key) >= 4:
                for bb in sorted(range(len(blocks)), key=lambda x: abs(x - b)):
                    hit = texts[bb][0].find(key)
                    if hit < 0 and len(key) > 12:   # the model's words may differ a little at the end
                        hit = texts[bb][0].find(key[:12])
                    if hit >= 0:
                        dest = (bb, sentence_start(blocks[bb][0], texts[bb][1][hit]) if hit else 0)
                        break
            if dest is None:
                dest = (b, sentence_start(chars, pos))
            if not note.get("at"):   # name the sentence, for a view with the notes after the text
                words = "".join(c for c, _ in blocks[dest[0]][0][dest[1]:dest[1] + 60]).split()[:3]
                if words:
                    note.set("at", " ".join(words))
            moves.append((b, i, dest, note))
    for b, (_, notes) in enumerate(blocks):
        notes.clear()
    for _, _, (bb, pos), note in moves:
        blocks[bb][1].append((pos, note))
    for _, notes in blocks:
        notes.sort(key=lambda x: x[0])


def cut_points(blocks, ocr: list[str]):
    """Per block {character position where a printed line ends: add a hyphen there}, and the
    share of the transcription's letters found in the OCR. A break inside a word of the model's
    text gets a hyphen, also where the OCR saw none (italics, a faint hyphen): without it the
    reader runs the two halves on with a space ("terr am", "comme morat": W05817 p. 8)."""
    T, where = [], []   # letters of the transcription, and (block, char) of each
    for b, (chars, _) in enumerate(blocks):
        for c, (ch, _) in enumerate(chars):
            for l in norm(ch):
                T.append(l)
                where.append((b, c))
    O, line = [], []
    for j, o in enumerate(ocr):
        O += o
        line += [j] * len(o)
    cuts = [{} for _ in blocks]
    if not T or not O:
        return cuts, 0.0
    sm = SequenceMatcher(None, "".join(T), "".join(O), autojunk=False)
    first, last, got = {}, {}, {}
    matched = 0
    for a, o, size in sm.get_matching_blocks():
        if size < MIN_BLOCK:
            continue
        matched += size
        for k in range(size):
            j = line[o + k]
            first.setdefault(j, (a + k, o + k))
            last[j] = (a + k, o + k)
            got[j] = got.get(j, 0) + 1
    starts = [0]
    for o in ocr:
        starts.append(starts[-1] + len(o))
    kept = []
    for j in sorted(got):
        if got[j] < max(MIN_BLOCK, LINE_FOUND * len(ocr[j])):
            continue
        if kept and first[j][0] <= last[kept[-1]][0]:
            if got[j] > got[kept[-1]]:
                kept.pop()   # out of order: keep the better aligned of the two
            else:
                continue
        kept.append(j)
    for a, b in zip(kept, kept[1:]):
        ta, oa = last[a]
        tb, ob = first[b]
        # where the line break falls, counting the letters the OCR has on either side unaligned
        lo = ta + 1 + (starts[a + 1] - 1 - oa)
        hi = tb - (ob - starts[b])
        target = min(max((lo + hi) / 2, ta + 1), tb)
        (ba, ca), (bb, cb) = where[ta], where[tb]
        if ba != bb:
            continue   # the break between two blocks is a line break anyway
        chars = blocks[ba][0]
        tc = where[min(int(round(target)), len(where) - 1)][1]
        cand = [c for c in range(ca + 1, cb + 1)
                if chars[c - 1][0] == " " or (chars[c - 1][0] == "-" and chars[c][0] != " ")]
        if cand:
            cuts[ba][min(cand, key=lambda c: abs(c - tc))] = False
        elif ca < tc <= cb and not (chars[tc - 1][0].isdigit() and chars[tc][0].isdigit()):
            # a word broken where the model left out the hyphen; a number never, and "Deut.28." or "est,gratias"
            # (no space after the stop) is no broken word: a hyphen only between two letters
            cuts[ba][tc] = chars[tc - 1][0].isalpha() and chars[tc][0].isalpha()
    return cuts, matched / len(T)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markup(chars) -> str:
    """Characters back to text with <bl> and <i> (inside it); a space has only the type of the letters on
    both sides of it."""
    out, open_ = [], []   # the open tags, outermost first
    for k, (c, it) in enumerate(chars):
        if c == " ":
            it = (BLACK * ("bl" in open_) | ITALIC * ("i" in open_)) & (chars[k + 1][1] if k + 1 < len(chars) else 0)
        want = ["bl"] * bool(it & BLACK) + ["i"] * bool(it & ITALIC)
        same = 0
        while same < min(len(open_), len(want)) and open_[same] == want[same]:
            same += 1
        out += [f"</{t}>" for t in reversed(open_[same:])] + [f"<{t}>" for t in want[same:]]
        open_ = want
        out.append(esc(c))
    out += [f"</{t}>" for t in reversed(open_)]
    s = "".join(out)
    s = re.sub(r" ((?:</(?:i|bl)>)+)", r"\1 ", s)
    return re.sub(r"((?:<(?:i|bl)>)+) ", r" \1", s)


def stray_blocks(page) -> list:
    """The blocks of a page outside its first <main_text>, which is what the reader shows: the model
    sometimes closes <main_text> at a heading and goes on under <page> or in a second <main_text>,
    or puts a heading in front of it. <l> under a <page> without <main_text> is no stray: title and
    contents pages are written so (build_reader.wrap_loose_lines)."""
    main = page.find("main_text")
    out = []
    for e in page:
        if e.tag in ("p", "head") or (e.tag == "l" and main is not None):
            out.append(e)
        elif e.tag == "main_text" and e is not main:
            out += [c for c in e if c.tag in ("p", "head", "l")]
    return out


def gather(page) -> None:
    """All the running text of a page in one <main_text>, in the order the model wrote it."""
    if not stray_blocks(page):
        return
    parts = [e for e in page if e.tag in ("main_text", "p", "head", "l", "note")]
    main = ET.Element("main_text")
    page.insert(list(page).index(parts[0]), main)
    for e in parts:
        page.remove(e)
        main.extend(list(e) if e.tag == "main_text" else [e])


def add_tail(parent, child) -> None:
    """Keep child's tail text in parent when child is taken out."""
    if not child.tail:
        return
    kids = list(parent)
    i = kids.index(child)
    if i:
        kids[i - 1].tail = (kids[i - 1].tail or "") + child.tail
    else:
        parent.text = (parent.text or "") + child.tail


DECORATION_KINDS = {"woodcut", "ornament", "initial"}
DECORATION_WORDS = re.compile(r"\binitial\b|\bornament|\bheadpiece|\btailpiece|\bhead-piece|\btail-piece|\bfleuron"
                              r"|printer'?s (device|mark)|\bvignette", re.I)


def is_decoration(kind: str | None, alt: str) -> bool:
    """A woodcut, a printer's ornament or a decorated initial: no picture for the reader (prompts/layout/
    _common.txt, rule B). The model is told so; what it still writes is dropped here."""
    return (kind or "").strip().lower() in DECORATION_KINDS or bool(DECORATION_WORDS.search(alt or ""))


def split_page(xml: str, ocr: list[tuple[str, bool]] | None) -> tuple[str, str | None]:
    """The model's page with <head>/<p> blocks as a page with numbered <l> lines (see the module
    doc). Returns (xml, warning). A page that already has <l> lines comes back as it is."""
    page = ET.fromstring(xml)
    gather(page)
    main = page.find("main_text")
    if main is None or not any(e.tag in ("p", "head") for e in main):
        return xml, None
    for e in list(main):   # a <figure> inside a paragraph goes right after it
        if e.tag in ("p", "head"):
            for f in [f for f in e if f.tag == "figure"]:
                add_tail(e, f)
                e.remove(f)
                f.tail = None
                main.insert(list(main).index(e) + 1, f)
    elems = [e for e in main if e.tag in ("p", "head", "l")]
    # a picture (<figure>) stays where it stands: after the lines of the block before it
    figures_after: dict[int, list] = {}
    k = -1
    for e in main:
        if e in elems:
            k = elems.index(e)
        elif e.tag == "figure" and not is_decoration(e.get("kind"), "".join(e.itertext())):
            figures_after.setdefault(k, []).append(e)
    blocks = [flatten(e) for e in elems]
    # a note the model put between the blocks goes with the first line of the block before it
    # (or after it, at the top of the page)
    b = -1
    for e in main:
        if e in elems:
            b = elems.index(e)
        elif e.tag == "note" and elems:
            n = copy.deepcopy(e)
            n.tail = None
            blocks[max(b, 0)][1].append((0, n))
    anchor_notes(blocks)
    if ocr:
        cuts, share = cut_points(blocks, [o[0] for o in ocr])
    else:
        cuts, share = [{} for _ in blocks], 0.0

    new_main = ET.Element("main_text")
    marg = ET.Element("marginalia")
    n = 0

    def figures(k):
        for f in figures_after.get(k, []):
            attrs = {"type": "figure", **{a: f.get(a) for a in ("kind", "place") if f.get(a)}}
            attrs["alt"] = " ".join("".join(f.itertext()).split())
            ET.SubElement(new_main, "l", attrs)

    figures(-1)
    for b_i, (e, (chars, notes), cs) in enumerate(zip(elems, blocks, cuts)):
        bounds = [0] + sorted(c for c in cs if 0 < c < len(chars)) + [len(chars)]
        first_n = n + 1
        line_of = []
        for s, t in zip(bounds, bounds[1:]):
            seg = chars[s:t]
            while seg and seg[0][0] == " ":
                seg = seg[1:]
            while seg and seg[-1][0] == " ":
                seg = seg[:-1]
            if not seg:
                continue
            if cs.get(t) and seg[-1][0] not in "-=":
                seg = seg + [("-", seg[-1][1])]
            n += 1
            l = ET.fromstring(f"<l>{markup(seg)}</l>")
            l.set("n", str(n))
            if e.tag == "head":
                l.set("type", "heading")
                l.set("level", e.get("level") or "2")
                if e.get("caput"):
                    l.set("caput", e.get("caput"))
                if n == first_n:   # where the heading starts: one printed over several lines stays one
                    l.set("para", "true")
            elif e.tag == "l":
                for k in ("type", "level", "caput", "para"):
                    if e.get(k):
                        l.set(k, e.get(k))
            else:
                if e.get("type"):   # a note below the text: <p type="foot">
                    l.set("type", e.get("type"))
                if n == first_n and e.get("cont") != "true":
                    l.set("para", "true")
            new_main.append(l)
            line_of.append((t, n))
        for pos, note in notes:
            ln = next((k for t, k in line_of if pos < t), line_of[-1][1] if line_of else n)
            note.set("line_from", str(ln))
            note.set("line_to", str(ln))
            marg.append(note)
        figures(b_i)

    i = list(page).index(main)
    page.remove(main)
    page.insert(i, new_main)
    if len(marg):
        page.insert(i + 1, marg)
    page.set("lines", "ocr" if ocr else "none")
    signature_line(page)
    warning = None
    if not ocr:
        warning = "no usable OCR layer: paragraphs could not be split into printed lines"
    elif share < WEAK:
        warning = f"only {share:.0%} of the letters found in the OCR layer: line breaks partly missing"
    if warning:
        w = ET.Element("warning", type="line_split")
        w.text = warning
        page.insert(0, w)
    ET.indent(page)
    return ET.tostring(page, encoding="unicode"), warning


def signature_line(page: ET.Element) -> str | None:
    """Take a gathering signature that was set as the last line of the running text ("A ııı" below
    "indicat, vi-": W05817 p. 7) out of it and into bottom_margin/signature, where the reader shows
    it as page furniture; in the text it glued itself to the broken word ("viA") and cut the word
    off from the rest of it on the next page. Returns the signature, or None."""
    main = page.find("main_text")
    ls = [e for e in (main if main is not None else []) if e.tag == "l"]
    if len(ls) < 2 or ls[-1].get("type") in ("heading", "figure", "diagram"):
        return None
    text = " ".join("".join(ls[-1].itertext()).split())
    if not is_signature(text):
        return None
    kids = list(main)
    i = kids.index(ls[-1])
    if i:
        kids[i - 1].tail = ls[-1].tail
    main.remove(ls[-1])
    bm = page.find("bottom_margin")
    if bm is None:
        bm = ET.Element("bottom_margin")
        bm.tail = main.tail
        after = page.find("marginalia")
        page.insert(list(page).index(after if after is not None else main) + 1, bm)
    sig = bm.find("signature")
    if sig is None:
        sig = ET.SubElement(bm, "signature")
    if not (sig.text or "").strip():
        sig.text = text
    return text


def is_signature(text: str) -> bool:
    """Is this short line a gathering signature? Over the whole library (fix_breaks.py --dry-run) a looser
    pattern also took years (MDCXII.), words (Qui, Dei, Si, vel, Ex), headings (TITVLI) and section
    numbers (§ 26.): so the letter is one letter, perhaps doubled (A, Aa, Bbb, not De or MDC), the leaf
    one digit or a Roman number of a signed leaf (ij-vij; I, j alone after a space), and a small letter
    needs a space or a digit after it (b ij, a 2, not si)."""
    m = SIGNATURE.fullmatch(" ".join(text.split()).translate(DOTLESS))
    if not m:
        return False
    mark, sp, leaf = m["mark"], m["sp"], m["leaf"]
    if mark.isalpha():
        if mark.islower() and not (sp or leaf.isdigit()):
            return False
    if leaf.isdigit() or leaf in LEAVES:
        return True
    if leaf in ("i", "j"):
        return bool(sp) and not mark.islower()
    return bool(sp) and leaf in ("II", "III", "IIII", "IV", "V", "VI")


def add_hyphen(line: ET.Element) -> None:
    """A hyphen after the last letter of a line, inside its italics if the line ends in them."""
    last = None

    def walk(e):
        nonlocal last
        if e.text and e.text.strip():
            last = (e, "text")
        for c in e:
            walk(c)
            if c.tail and c.tail.strip():
                last = (c, "tail")

    walk(line)
    if last:
        v = getattr(*last)
        setattr(*last, v.rstrip() + "-" + v[len(v.rstrip()):])


def drop_hyphen(line: ET.Element) -> None:
    """Take the hyphen off the end of a line (add_hyphen undone)."""
    last = None

    def walk(e):
        nonlocal last
        if e.text and e.text.strip():
            last = (e, "text")
        for c in e:
            walk(c)
            if c.tail and c.tail.strip():
                last = (c, "tail")

    walk(line)
    if last:
        v = getattr(*last)
        k = len(v.rstrip())
        if k and v[k - 1] == "-":
            setattr(*last, v[:k - 1] + v[k:])


def rehyphen(page: ET.Element, para_xml: str) -> tuple[list[ET.Element], list[ET.Element]]:
    """Mend the line ends of a page split before cut_points hyphenated as it does now, from the model's
    text (pNNNN.para.xml): a line that ends inside a word gets its hyphen, and a hyphen that is not in
    the model's text and does not stand between two letters goes ("v.67.-", "Malim,-": a break after
    a stop the model wrote without a space). The lines are laid over the model's blocks in order,
    letter for letter; when they are not its text (a page changed since) nothing is done. Returns
    (the lines that got a hyphen, the lines that lost one)."""
    pp = ET.fromstring(para_xml)
    gather(pp)
    main = pp.find("main_text")
    if main is None:
        return [], []
    texts = []
    for e in main:
        if e.tag in ("p", "head", "l"):
            for f in [f for f in e if f.tag == "figure"]:   # as split_page: a picture is no text
                add_tail(e, f)
                e.remove(f)
            t = "".join(c for c, _ in flatten(e)[0]).strip()
            if t:
                texts.append(t)
    b, pos, broken, wrong = 0, 0, [], []
    for l in page.iterfind("main_text/l"):
        t = " ".join("".join(l.itertext()).split())
        if not t or l.get("type") == "figure":
            continue
        while b < len(texts) and pos >= len(texts[b]):
            b, pos = b + 1, 0
        if b == len(texts):
            return [], []
        s = texts[b]
        if s.startswith(t, pos):   # the model's text as it is: a break between two letters lacks its hyphen
            end = pos + len(t)
            if 0 < end < len(s) and s[end - 1].isalpha() and s[end].isalpha():
                broken.append(l)
        elif t.endswith("-") and s.startswith(t[:-1], pos):   # a hyphen of the split, not of the model
            end = pos + len(t) - 1
            if not (0 < end < len(s) and s[end - 1].isalpha() and s[end].isalpha()):
                wrong.append(l)
        else:
            return [], []
        pos = end + (end < len(s) and s[end] == " ")
    if b < len(texts) - 1 or (b < len(texts) and pos < len(texts[b])):
        return [], []
    for l in broken:
        add_hyphen(l)
    for l in wrong:
        drop_hyphen(l)
    return broken, wrong


def mark_heads(page: ET.Element, para_xml: str) -> int:
    """Give the first line of every heading para="true" on a page split before split_page did so,
    from the model's version of the page (pNNNN.para.xml): its <head>s in order are laid over the
    page's heading lines by their letters. Returns the number of lines marked."""
    heads = [lineboxes.norm("".join(h.itertext())) for h in ET.fromstring(para_xml).iterfind("main_text/head")]
    lines = [l for l in page.iterfind("main_text/l") if l.get("type") == "heading"]
    marked, h, got = 0, -1, ""
    for l in lines:
        text = lineboxes.norm("".join(l.itertext()))
        if not text and h >= 0:   # a number or a rule only
            continue
        # this line goes on the current heading while that has letters left that it starts with
        rest = heads[h][len(got):] if 0 <= h < len(heads) else ""
        if rest and text and (rest.startswith(text) or SequenceMatcher(None, rest[:len(text)], text).ratio() >= 0.8):
            got += text
            continue
        h, got = h + 1, text
        if l.get("para") != "true":
            l.set("para", "true")
            marked += 1
    return marked


def main():
    import pymupdf
    sys.stdout.reconfigure(encoding="utf-8")
    para, pdf, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    ocr = printed_lines(pymupdf.open(pdf)[n - 1])
    xml, warning = split_page(para.read_text(encoding="utf-8"), ocr)
    print(xml)
    if warning:
        print("WARNING:", warning)


if __name__ == "__main__":
    main()
