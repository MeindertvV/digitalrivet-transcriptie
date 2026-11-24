"""A book as a file: book.xml, as transcribe.py and ocr_naar_xml.py write it, and book.json made from it; and the
book as the later stages read it, with what the reader corrected (load(), below).

The writing functions stood in transcription/transcribe.py, and every script that writes a book took them from
there, loading the whole transcription (and the model's library) to do so. No model and no PDF is needed for them.
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from common.notice import blank_notice
from common.paths import PILOT
from common.text import bare_letters

INLINE = ("i", "hi", "bl")   # type within a line: italics, black letter (Fraktur)


def letters(s: str) -> str:
    """Lower-case letters only, with the spellings OCR and transcription differ on made equal
    (long s read as f, u/v, i/j, æ/œ)."""
    s = s.lower().replace("ſ", "s").replace("f", "s").replace("v", "u").replace("j", "i")
    s = s.replace("æ", "ae").replace("œ", "oe")
    return "".join(c for c in s if c.isalpha())


def flag_merged_lines(page) -> str | None:
    """Mark lines the model merged into paragraphs: each gets merged="true" and the
    page gets a <warning type="merged_lines">. Returns the warning text, if any."""
    lines = list(page.iter("l"))
    lengths = sorted(len(l.text or "") for l in lines)
    if not lengths:
        return None
    median = lengths[len(lengths) // 2]
    limit = max(150, 3 * median)
    if lengths[-1] <= limit:
        return None
    merged = [l for l in lines if len(l.text or "") > limit]
    for l in merged:
        l.set("merged", "true")
    text = (f"{len(merged)} line(s) contain several printed lines merged together "
            f"(longest {lengths[-1]} chars, median {median}); line numbering is unreliable")
    warning = ET.Element("warning", type="merged_lines")
    warning.text = text
    warning.tail = "\n  "
    page.insert(0, warning)
    return text


def mark_numbered(book: ET.Element):
    """Numbered theses whose indentation the text layer missed: a line starting with the
    next number in the sequence (or 1-2 after a sentence end, a new disputation) gets para.
    Running through the whole book catches theses that start at the top of a page; the
    sequence keeps verse numbers at line starts ("Matth. / 20. Quorum") out."""
    last, prev = 0, ""
    for l in book.iterfind("page/main_text/l"):
        text = "".join(l.itertext()).strip()
        m = re.match(r"(\d{1,3})\.?\s+[A-ZÆŒ]", text)
        if m and l.get("type") != "heading":
            k = int(m.group(1))
            if l.get("para") == "true" or k == last + 1 or (k <= 2 and re.search(r"[.:?!]\W*$", prev)):
                l.set("para", "true")
                last = k
        prev = text


def line_markup(el: ET.Element) -> str:
    """Text of a line with <i>/<hi>/<bl> kept as tags, whitespace normalised."""
    def inner(e):
        out = e.text or ""
        for c in e:
            t = inner(c)
            out += f"<{c.tag}>{t}</{c.tag}>" if c.tag in INLINE and t.strip() else t
            out += c.tail or ""
        return out
    text = " ".join(inner(el).split())
    return re.sub(r"<(i|hi|bl)>\s*</\1>", "", text)


def book_json(book: ET.Element) -> dict:
    """book.xml as JSON for a frontend: per page its lines (italics as inline <i>/<hi> tags,
    heading level 0 for running text, para on the first line of a paragraph), margin notes
    and page furniture. Empty values are left out."""
    def text(el, path):
        return " ".join("".join(el.find(path).itertext()).split()) if el.find(path) is not None else ""

    pages = []
    for pg in book.iter("page"):
        lines = []
        for l in pg.iterfind("main_text/l"):
            d = {"n": int(l.get("n")) if (l.get("n") or "").isdigit() else None, "text": line_markup(l),
                 "heading": int(l.get("level") or 2) if l.get("type") == "heading" else 0}
            if l.get("caput"):
                d["caput"] = l.get("caput")
            if l.get("para") == "true":
                d["para"] = True
            if l.get("merged") == "true":
                d["merged"] = True
            lines.append(d)
        notes = [{**{k: int(v) if k.startswith("line_") and v.isdigit() else v for k, v in nt.attrib.items()},
                  "text": " ".join("".join(nt.itertext()).split())} for nt in pg.iterfind("marginalia/note")]
        page = {"n": int(pg.get("n")), "type": pg.get("type") or "text", "model": pg.get("model"),
                "number": text(pg, "page_number"), "running_head": text(pg, "running_head"),
                "warnings": [{"type": w.get("type"), "text": w.text} for w in pg.iterfind("warning")],
                "lines": lines, "marginalia": notes,
                "catchword": text(pg, "bottom_margin/catchword"), "signature": text(pg, "bottom_margin/signature"),
                "notes": text(pg, "notes")}
        pages.append({k: v for k, v in page.items() if v not in ("", [], None)})
    return {"source": book.get("source"), "pages": pages}


def write(book: ET.Element, folder: Path):
    """Write book.xml and the book.json made from it into folder."""
    ET.ElementTree(book).write(folder / "book.xml", encoding="utf-8", xml_declaration=True)
    (folder / "book.json").write_text(json.dumps(book_json(book), ensure_ascii=False, indent=1), encoding="utf-8")


# ================================================================ reading: a book as the later stages read it
# This was enrich/book.py (REFACTOR_PLAN.md step 63), line for line but for two things: its `letters` was
# common/text.bare_letters (the letters above are another function), and lxml is imported where it is used, so that
# what only writes a book does not need it (common/units.py takes line_markup from here, and some scripts that load
# it have a Python of their own).
#
# book.xml itself stays as transcribed. The contents (structure.py) and the site (sitebuild/edition/build_reader.py) both
# start from load(), so that a line number means the same in both.

EDIT_FILE = "book_edit.json"   # in pilot/<work>/: the lines the reader corrected (apply_suggestions.py)

# was toc_haiku.lines_of, then headings.lines_of (REFACTOR_PLAN.md step 63): the <l> lines of a page's main text
def lines_of(page) -> list:
    main = page.find("main_text")
    return [l for l in main if l.tag == "l"] if main is not None else []


# ---------------------------------------------------------------- edits from the reader

IT = chr(42)   # the reader's italics mark (in code: a * in a source line reads as a change)


def edit_src(text: str) -> str:
    """A corrected line as book.xml holds it, so that inline_toks makes it appear as the reader made
    it: the reader's italics (IT ... IT) is book.xml's <i>...</i>, and a star behind a backslash is a
    star."""
    out, ital, i = [], False, 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):   # \* is a star, \\ a backslash
            out.append("&#42;" if text[i + 1] == IT else text[i + 1])
            i += 2
            continue
        if c == IT:   # italics on, or off again
            out.append("<i>" if not ital else "</i>")
            ital = not ital
        else:
            out.append(c)
        i += 1
    return "".join(out) + ("</i>" if ital else "")


def read_edits(work: str | None) -> tuple[dict, dict, list]:
    """pilot/<work>/book_edit.json (apply_suggestions.py): {(scan, printed line): corrected line},
    {lang: {unit id or position: translated text}}, and the paragraph breaks and joins (a line
    starts a paragraph of its own, or goes on in the line before it).

    Missing or unreadable, all are empty and the book is built as it is; the build still depends
    on the file (fingerprint), so a new edit builds the book again."""
    if not work:
        return {}, {}, []
    path = PILOT / work / EDIT_FILE
    if not path.is_file():
        return {}, {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"  {path.name} not read ({e}): the book is built as it is")
        return {}, {}, []
    lines = {}
    for e in data.get("lines") or []:
        try:
            lines[(int(e["scan"]), str(e["line"]))] = str(e["after"])
        except (KeyError, TypeError, ValueError):
            continue
    return lines, data.get("translation") or {}, list(data.get("paragraphs") or [])


def last_words(text: str, n: int = 4) -> str:
    """The words a sentence of the text ends with (what tells a line's place apart)."""
    words = re.findall(r"[\w\u0080-\uffff]+", text or "")
    return " ".join(words[-n:]).lower()


def apply_paragraphs(root, changes: list) -> int:
    """The paragraph breaks made and the lines joined in the reader, in book.xml in place.

    A line with <para>true</para> is the first of a paragraph (ocr_split.py, mark_heads.py, and the
    reader); one without it goes on in the line before. So a place that holds the change is marked
    where the break is (or is not) as taken. A break is taken where it is: a paragraph start whose
    line's last words are the place's; a join where a line that is not a start goes on in the line
    the place names (and that line is the one before it in the transcription). A place not found is
    reported, never put elsewhere."""
    if not changes:
        return 0
    n, missed = 0, []
    for c in changes:
        op = c.get("op")
        head = (c.get("head") or {}).get("scan")
        head_tail = last_words((c.get("head") or {}).get("text") or "")
        tail = last_words(c.get("text") or "") if op == "split" else ""
        hit = None
        for pg in root.iter("page"):
            for i, l in enumerate(pg.iterfind("main_text/l")):
                ls = pg.findall("main_text/l")
                is_start = l.get("para") == "true"
                if op == "split" and not is_start and tail and last_words(" ".join(l.itertext())).endswith(tail):
                    # the place of the break: a paragraph starts here, and should not
                    prev = ls[i - 1] if i else None
                    if prev is not None and (not head_tail or last_words(" ".join(prev.itertext())).endswith(head_tail)):
                        hit = (l, True)
                        break
                if op == "join" and is_start and head_tail and last_words(" ".join(l.itertext())).endswith(head_tail):
                    # the place of the join: this line goes on in the one after it
                    nxt = ls[i + 1] if i + 1 < len(ls) else None
                    if nxt is not None and nxt.get("para") != "true":
                        hit = (nxt, False)
                        break
            if hit:
                break
        if not hit:
            missed.append(f"{c.get('scan')}.{c.get('line')} ({op})")
            continue
        el, start = hit
        if start:
            el.set("para", "true")
        elif el.get("para") == "true":
            del el.attrib["para"]
        n += 1
    if missed:
        print(f"  paragraph changes not found: {', '.join(missed)}")
    if n:
        print(f"  {n} paragraph changes as made in the reader")
    return n


def wrap_loose_lines(root) -> int:
    """Title and contents pages carry their printed lines as <l> straight under <page> (the prompt asks
    for that); the reader shows what is in <main_text>, so those lines go in one."""
    from lxml import etree
    n = 0
    for pg in root.iter("page"):
        loose = pg.findall("l")
        if loose and pg.find("main_text") is None:
            main = etree.Element("main_text")
            loose[0].addprevious(main)
            for l in loose:
                main.append(l)
            n += 1
    return n


def apply_line_edits(root, edits: dict) -> int:
    """book.xml's lines as the reader corrected them, in place: <l n="4">3, 4 atque 5</l> or
    <l n="4"><i>Deo</i> laus</l>. A line that is no longer the one the correction was made on
    (book.xml changed since: the same letters are not there any more) keeps what book.xml has."""
    if not edits:
        return 0
    n = 0
    for pg in root.iter("page"):
        idx = int(pg.get("n"))
        for l in pg.iterfind("main_text/l"):
            text = edits.get((idx, str(l.get("n"))))
            if text is None:
                continue
            was = " ".join(l.itertext())
            done = fit_and_set(l, text, was)
            if done:
                n += 1
    if n:
        print(f"  {n} lines as corrected in the reader")
    return n


def fit_and_set(l, text: str, was: str) -> bool:
    """Line l as the reader corrected it (text), if it still is the line the correction was made
    on (was). True when it was set."""
    from lxml import html as lhtml
    plain_text = IT.join(text.split(IT))
    if plain_text == was:
        return False   # as it is: nothing to change (no italics added)
    if bare_letters(plain_text) != bare_letters(was):
        return False   # not the line the correction was made on any more: book.xml's stands
    plain = len(l) == 0   # what book.xml has is all plain text: so is what comes back
    frag = lhtml.fragment_fromstring(edit_src(text), create_parent="div")
    keep = dict(l.attrib)
    l.clear()
    l.text = frag.text
    for c in frag:
        l.append(c)
    l.attrib.update(keep)
    if not plain:
        l.set("edited", "true")   # (with an <i> of the reader's in it: it may end in one)
    return True


def apply_translation_edits(tr: dict, edits: dict) -> int:
    """translation_<lang>.json's units as the reader corrected them, in place. The reader names a
    passage by its unit id and by its position among the units, whichever still holds. The corrected
    text comes as the reader holds it: its *…* are italics, its \n the translation's line breaks."""
    if not edits:
        return 0
    n = 0
    units = tr.get("units") or []
    for key, text in edits.items():
        u = next((u for u in units if str(u.get("id")) == str(key)), None)
        if u is None and str(key).isdigit() and int(key) < len(units):
            u = units[int(key)]
        if u is None:
            continue
        text = re.sub(r"(?<!\\)\*([^*]*)(?<!\\)\*", r"<i>\1</i>", text.replace("\\*", "*")).replace("\n", " ")
        if " ".join(str(u.get("tr") or "").split()) == " ".join(text.split()):
            continue
        u["tr"], u["edited"] = text, True
        n += 1
    if n:
        print(f"  {n} translated passages as corrected in the reader")
    return n


# ---------------------------------------------------------------- transcription


def load(book_dir: Path, work: str | None):
    """The book's root, its reader's edits applied, and what they say of the translations and the
    scanner's cover sheet: (root, translation edits, cover scans)."""
    from lxml import etree
    root = etree.parse(str(book_dir / "book.xml"), etree.XMLParser(recover=True)).getroot()
    wrap_loose_lines(root)
    notice = blank_notice(root)   # the scanner's cover sheet is not part of the book
    line_edits, tr_edits, para_edits = read_edits(work)
    apply_line_edits(root, line_edits)
    apply_paragraphs(root, para_edits)
    return root, tr_edits, notice
