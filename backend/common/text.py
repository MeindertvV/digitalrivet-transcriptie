"""Making texts comparable: folding spellings, letters only, Roman numerals (REFACTOR_PLAN.md steps 56 and 57).

Only what does the same is one function here. Two functions that tests/test_text_functions.py finds giving the same
on its 200 lines are one only when their code is the same too: translate.plain and loci.clean agree on those lines,
but loci.clean also cuts a text off at 160 letters, so they stay two. The functions that differ are here each under
a name that says what it is for, with the place it came from ("# was ..."); that place keeps its old name for it.
Not here: the letters of common/units.py and common/bookfile.py, which
are in common/ already.
"""
import hashlib
import re
import unicodedata


def element_text(el) -> str:
    """The text of an XML element, with that of its children, its white space made single spaces:
    <l>De  <i>Deo</i>\\n uno</l> is "De Deo uno". Was enrich/book.plain and enrich/register.plain."""
    return " ".join("".join(el.itertext()).split())


# --- Letters for pairing a transcribed line with the OCR on its scan (pdfboxes.py, ocr_split.py, parts.py)

# was pdfboxes.norm
def pairing_letters(s: str) -> str:
    """Letters only, with the differences between this OCR and the transcription folded away:
    long s (read as f), u/v, i/j, ligatures."""
    s = s.lower()
    for a, b in (("ſ", "s"), ("æ", "ae"), ("œ", "oe"), ("ß", "ss"), ("&", "et"), ("ꝗ", "q"), ("ę", "e")):
        s = s.replace(a, b)
    s = re.sub(r"[\W\d_]+", "", s)
    return s.replace("f", "s").replace("v", "u").replace("j", "i")


# was ocr_split.norm
def pairing_letters_and_digits(s: str) -> str:
    """pairing_letters, but keeping digits: chapter and verse numbers help the alignment."""
    return "".join(c if c.isdigit() else pairing_letters(c) for c in s)


# was parts.letters
def pairing_letter_count(text: str) -> int:
    return sum(len(pairing_letters(c)) for c in text)


# --- The letters a block of the translation is known by (translation/blocks.py)

# was blocks.words
def block_words(text: str) -> list[str]:
    """The words of a text, whatever the markup, long s or u/v and i/j of this reading."""
    text = re.sub(r"<[^>]+>", "", text).lower().replace("ſ", "s").replace("v", "u").replace("j", "i")
    return re.findall(r"[^\W\d_]+", text)


# was blocks.letters
def block_letters(text: str) -> str:
    return "".join(block_words(text))


# was blocks.sha
def block_key(text: str) -> str:
    return hashlib.sha1(block_letters(text).encode()).hexdigest()[:12]


# --- Printed page numbers and headings as the contents lists read them (enrich/book.py)
NONE_PRINTED = ("", "none", "none printed")


# was book.clean
def unless_none(s):
    s = (s or "").strip()
    return "" if s.strip("[]").lower() in NONE_PRINTED else s


# was book.letters
def bare_letters(s: str) -> str:
    """Lower-case letters only, so hyphenation and spacing do not matter when comparing."""
    return re.sub(r"[\W\d_]+", "", s.lower())


# --- Bible references (references/bibleref.py)

class _BibleFold(dict):
    """str.translate table of bible_fold(), filled one character at a time."""

    def __missing__(self, o):
        c = chr(o).lower()
        if c in "æœ":
            c = "e"
        elif c == "ſ":
            c = "s"
        else:
            c = unicodedata.normalize("NFKD", c)[:1] or c
        c = {"j": "i", "v": "u"}.get(c, c)
        self[o] = c if len(c) == 1 else chr(o)
        return self[o]


_BIBLE_FOLD = _BibleFold()


# was bibleref.norm
def bible_fold(s: str) -> str:
    """Lower case, no accents, j->i, v->u, long s->s, ae/oe ligatures as one letter: the same length as s,
    so a match in bible_fold(s) is at the same place in s."""
    return s.translate(_BIBLE_FOLD)


# --- Titles in the catalogue (subjects/classify.py)

# was classify.norm
def title_fold(s: str) -> str:
    s = s.replace("æ", "ae").replace("Æ", "Ae").replace("œ", "oe").replace("Œ", "Oe").replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.replace("ae", "e").replace("oe", "e").replace("u", "v").replace("j", "i")


_TITLE_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


# was classify.roman
def roman_in_title(s: str) -> int | None:
    s = s.upper().replace("U", "V")
    if s == "PRIMVM":
        return 1
    if not s or any(c not in _TITLE_ROMAN for c in s):
        return None
    n = 0
    for a, b in zip(s, s[1:] + " "):
        v = _TITLE_ROMAN[a]
        n += -v if b in _TITLE_ROMAN and _TITLE_ROMAN[b] > v else v
    return n


# --- Authors quoted in the text (references/auctoritates.py)

class _AuthorFold(dict):
    """str.translate table of author_fold(): one character for one."""

    def __missing__(self, o):
        c = chr(o)
        if c in "jJvVſæœÆŒ":
            r = {"j": "i", "J": "I", "v": "u", "V": "U", "ſ": "s", "æ": "e", "œ": "e", "Æ": "E", "Œ": "E"}[c]
        else:
            d = unicodedata.normalize("NFKD", c)
            r = d[0] if d else c
        self[o] = r
        return r


_AUTHOR_FOLD = _AuthorFold()


# was auctoritates.fold
def author_fold(s: str) -> str:
    """j->i, v->u, long s->s, æ/œ->e, accents off; the case kept and the same length as s."""
    return s.translate(_AUTHOR_FOLD)


_LOWER_ROMAN = {"i": 1, "u": 5, "x": 10, "l": 50, "c": 100}


# was auctoritates.roman
def roman_lowercase(s: str) -> int:
    n, prev = 0, 0
    for c in reversed(s):
        v = _LOWER_ROMAN[c]
        n, prev = (n - v, prev) if v < prev else (n + v, v)
    return n


# --- Headings as Haiku is shown them (subjects/loci.py)

# was loci.clean
def heading_snippet(markup: str, n: int = 160) -> str:
    from common.units import plain
    s = " ".join(plain(markup).split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


# --- The OCR text of a line and a printed page number (ocr/ocr_naar_xml.py)
_SPECK = re.compile(r"[|_—–~^*•¦‘’'\"`«»°¬=+<>\\/]+")
_PAGE_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


# was ocr_naar_xml.roman
def page_number(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    s = s.upper()
    if not s or any(c not in _PAGE_ROMAN for c in s):
        return None
    vals = [_PAGE_ROMAN[c] for c in s]
    return sum(-v if i + 1 < len(vals) and v < vals[i + 1] else v for i, v in enumerate(vals))


# was ocr_naar_xml.clean
def ocr_line(t: str) -> str:
    """OCR text of a line with whitespace normalised; ¬ (the OCR's hyphen) as -, control
    characters out, and specks at either end (a lone | _ — from the edge of the scan or the
    gutter, or a lone . : at the start). The words themselves are not changed."""
    t = t.replace("¬", "-").replace("\u00ad", "-")
    t = "".join(c for c in t if c >= " " or c == "\t")
    toks = t.split()
    while toks and (_SPECK.fullmatch(toks[0]) or toks[0] in (".", ":", ",", ";")):
        toks.pop(0)
    while toks and _SPECK.fullmatch(toks[-1]):
        toks.pop()
    return " ".join(toks)


# --- The catalogue and its text search (sitebuild/build_frontpage.py)

# was build_frontpage.norm
def catalogue_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return s.replace("ae", "e").replace("oe", "e").replace("u", "v").replace("j", "i")


# was build_frontpage.text_norm
def fulltext_fold(s: str) -> str:
    s = s.lower().replace("\u00ad", "").replace("æ", "ae").replace("ę", "ae").replace("œ", "oe").replace("ſ", "s")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not "̀" <= c <= "ͯ")


# --- The fragments of the text search (sitebuild/build_fragments.py)

# was build_fragments.norm
def fragment_fold(text: str) -> str:
    text = text.lower().replace("\u00ad", "").replace("æ", "ae").replace("ę", "ae").replace("œ", "oe").replace("ſ", "s")
    return "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))


# was build_fragments.fold
def search_fold(word: str) -> str:
    """The Latin in any of the printers' spellings, as the site's search folds it (foldLa in assets/reader.js):
    oe as ae, j as i, v as u. Letter for letter, after fragment_fold()."""
    return word.replace("oe", "ae").replace("j", "i").replace("v", "u")


# --- The words of the register, as the dictionary is matched (sitebuild/build_register.py)
_GREEK_IN_LATIN = {"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e", "θ": "th", "ι": "i",
                   "κ": "c", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
                   "ς": "s", "τ": "t", "υ": "y", "φ": "ph", "χ": "ch", "ψ": "ps", "ω": "o"}
_DICTIONARY_WORD = re.compile(r"[a-z]+")


class _DictionaryLetters(dict):
    """str.translate table of dictionary_words(): Greek letters in Latin ones, æ/œ e, long s s, other non-ASCII away."""

    def __missing__(self, o):
        c = chr(o)
        self[o] = _GREEK_IN_LATIN.get(c) or ("e" if c in "æœ" else "s" if c == "ſ" else c if c.isascii() else "")
        return self[o]


_DICTIONARY_LETTERS = _DictionaryLetters()
_ROUGH = re.compile("(?<!\\w)([αεηιουωaeiouy])\u0314")   # ἁ: the rough breathing follows its vowel


# was build_register.words
def dictionary_words(text: str) -> list[str]:
    """The words of a text as the dictionary is matched: lower case, no accents, Greek in Latin letters (with
    the rough breathing as h), j->i, v->u, k->c, long s->s, ae/oe/ai->e, ou->u, ph/th/ch kept."""
    s = _ROUGH.sub(r"h\1", unicodedata.normalize("NFKD", text.lower())).translate(_DICTIONARY_LETTERS)
    s = s.replace("j", "i").replace("v", "u").replace("k", "c")
    s = s.replace("ae", "e").replace("oe", "e").replace("ai", "e").replace("ou", "u")
    return _DICTIONARY_WORD.findall(s)


# --- Headings and their numbers (contents/headings.py, contents/structure.py)

# was headings.folded
def heading_fold(s: str) -> str:
    """Lower case letters only, u for v and i for j and s for the long s read as f."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if c.isalpha())
    return s.replace("v", "u").replace("j", "i").replace("f", "s")


_TENS = (("undecim", 11), ("duodecim", 12), ("decim", 10), ("uigesim", 20), ("uicesim", 20), ("tricesim", 30), ("trigesim", 30), ("quadragesim", 40), ("quinquagesim", 50))
_ORDINALS = (("unic", 1), ("prim", 1), ("secund", 2), ("alter", 2), ("terti", 3),
             ("quart", 4), ("quint", 5), ("sext", 6), ("septim", 7), ("octau", 8), ("nono", 9), ("nonu", 9), ("nona", 9))


# was headings.ordinal
def ordinal_compound(word: str) -> int | None:
    """The number in a written-out Latin ordinal, also compound: vigesimaprima is 21,
    decimusquartus 14."""
    for stem, base in _TENS:
        if word.startswith(stem):
            rest = re.sub(r"^(?:us|um|a|o|i|ae)", "", word[len(stem):])
            return base + (ordinal_compound(rest) or 0 if rest else 0)
    return next((v for stem, v in _ORDINALS if word.startswith(stem)), None)


# was headings.roman
def roman_any(s: str) -> int | None:
    """The value of a roman numeral, also as printed then (XIIII, XIIX)."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if not s or any(c not in vals for c in s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + "I"):
        total += -vals[a] if vals[a] < vals[b] else vals[a]
    return total


_WORD_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


# was structure.roman
def roman_word(w: str):
    """Value of a roman numeral below 400 (leniently written: IIII, XIIX too), or None; so the words
    Mi(nist.), Civil(is) and Dic(itur) stay words."""
    if not re.fullmatch(r"c{0,3}(?:xc|xl|l?x{0,4})(?:ix|iix|iv|v?i{0,4})", w) or not w:
        return None
    vals = [_WORD_ROMAN[c] for c in w]
    return sum(-v if i + 1 < len(vals) and v < vals[i + 1] else v for i, v in enumerate(vals))


_ORDINAL_STEMS = ["prim", "secund", "terti", "quart", "quint", "sext", "septim", "octau", "non", "decim",
                  "undecim", "duodecim"]
_ORDINAL_WORD = re.compile(r"(%s)(?:us|a|um)" % "|".join(sorted(_ORDINAL_STEMS, key=len, reverse=True)))


# was structure.ordinal
def ordinal_word(w: str):
    """3 for "tertius", "tertivs", "tertia"; None for any other word (u/v and i/j as in the heads)."""
    m = _ORDINAL_WORD.fullmatch(w.replace("v", "u").replace("j", "i"))
    return _ORDINAL_STEMS.index(m.group(1)) + 1 if m else None


_OPERA_WORD = re.compile(r"[^\W\d_]+|&", re.U)
_GREEK_HEBREW = re.compile(r"[Ͱ-Ͽἀ-῿֐-׿]")


# was ocnorm.nasal
def latin_nasal(w: str) -> str:
    """A vowel with a bar or tilde stands for the vowel and m or n: eū, quū, nō, tractādis -> eum, quum, non, tractandis."""
    if w.lower() in ("nō", "nõ", "nó"):
        return "non"
    w = unicodedata.normalize("NFC", w)
    return re.sub(r"[āēīōūãẽĩõũ](?=([pbm]|$)?)", lambda m: m.group(0) + ("n" if m.group(1) is None else "m"), w)


# was ocnorm.plain
def spelling_plain(w: str) -> str:
    """Lower case, no accents or ligatures: quæ, quàm, eū -> quae, quam, eu."""
    w = w.lower().replace("æ", "ae").replace("œ", "oe").replace("ß", "ss").replace("ſ", "s")
    return "".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))


# was ocnorm.key
def spelling_key(w: str) -> str:
    """The form two spellings of one word share: u/v, i/j, y/i, ae/e, doubled letters, ti/ci, quicquid/quidquid."""
    if w == "&":
        return "et"
    w = spelling_plain(latin_nasal(w)).replace("j", "i").replace("v", "u").replace("y", "i")
    w = w.replace("ae", "e").replace("oe", "e").replace("ph", "f").replace("th", "t")
    w = re.sub(r"(.)\1", r"\1", w).replace("cqu", "dqu").replace("nqu", "mqu")
    w = re.sub(r"^char", "car", w)
    w = re.sub(r"^circun|^ueruntamen", lambda m: m.group().replace("n", "m", 1) if m.group()[0] == "c" else "uerumtamen", w)
    w = re.sub(r"^quan(?=ui|diu)", "quam", w)
    w = "cum" if w == "qum" else w                  # the editors' quum
    w = w.replace("exs", "ex").replace("chr", "cr").replace("qut", "cut")
    return re.sub(r"ci(?=[aeou])", "ti", w)


# was ocnorm.words
def opera_words(text: str) -> list:
    """The words of a text, Greek and Hebrew left out: the Opera's OCR garbles them."""
    text = re.sub(r"&\s*c\b\.?", " etc ", re.sub(r"\[(\?|illegible[^\]]*)\]", "", text))
    return [w for w in _OPERA_WORD.findall(text) if not _GREEK_HEBREW.search(w)]
