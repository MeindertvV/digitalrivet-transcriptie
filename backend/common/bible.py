"""The books of the Bible, as the reader names them, with their chapters: one list for the genres
(subjects/genre_haiku.py: what Haiku may answer), the references (references/bibleref.py) and the register.
The list stood in both, the same 66 in the same order (REFACTOR_PLAN.md step 69).

build_frontpage.BIBLE_BOOKS is not this list: it finds a book in a catalogue title, by a pattern, and has 56
names (Samuel, Koningen, Korintiers ... without their number).
"""

import re

# book -> number of chapters (Joel and Malachi as the Vulgate counts them)
CHAPTERS = {
    "Genesis": 50, "Exodus": 40, "Leviticus": 27, "Numeri": 36, "Deuteronomium": 34, "Jozua": 24, "Richteren": 21,
    "Ruth": 4, "1 Samuel": 31, "2 Samuel": 24, "1 Koningen": 22, "2 Koningen": 25, "1 Kronieken": 29,
    "2 Kronieken": 36, "Ezra": 10, "Nehemia": 13, "Esther": 10, "Job": 42, "Psalmen": 150, "Spreuken": 31,
    "Prediker": 12, "Hooglied": 8, "Jesaja": 66, "Jeremia": 52, "Klaagliederen": 5, "Ezechiel": 48, "Daniel": 14,
    "Hosea": 14, "Joel": 4, "Amos": 9, "Obadja": 1, "Jona": 4, "Micha": 7, "Nahum": 3, "Habakuk": 3, "Sefanja": 3,
    "Haggai": 2, "Zacharia": 14, "Maleachi": 4, "Mattheus": 28, "Marcus": 16, "Lucas": 24, "Johannes": 21,
    "Handelingen": 28, "Romeinen": 16, "1 Korintiers": 16, "2 Korintiers": 13, "Galaten": 6, "Efeziers": 6,
    "Filippenzen": 4, "Kolossenzen": 4, "1 Tessalonicenzen": 5, "2 Tessalonicenzen": 3, "1 Timotheus": 6,
    "2 Timotheus": 4, "Titus": 3, "Filemon": 1, "Hebreeen": 13, "Jakobus": 5, "1 Petrus": 5, "2 Petrus": 3,
    "1 Johannes": 5, "2 Johannes": 1, "3 Johannes": 1, "Judas": 1, "Openbaring": 22,
}
BOOKS = list(CHAPTERS)   # the 66 books in canon order (Old Testament, 39; New Testament, 27), as the reader names them


# ---------------------------------------------------------------- a book as a model or a reader writes it

# other names of the unnumbered books (English, Latin, Dutch with accents), on the folded name (see _fold), by commas
_ALIASES = {
    "Genesis": "genesin, bereshit", "Exodus": "exodi", "Leviticus": "levitici", "Numeri": "numbers, numerorum",
    "Deuteronomium": "deuteronomy", "Jozua": "joshua, josua, iosue, iosua, josue",
    "Richteren": "judges, iudicum, judicum, liber iudicum", "Ruth": "", "Ezra": "esdras, esdrae",
    "Nehemia": "nehemiah, nehemias", "Esther": "ester", "Job": "iob, hiob, iobi",
    "Psalmen": "psalms, psalm, psalmi, psalter, psalterium", "Spreuken": "proverbs, proverbia, proverbiorum",
    "Prediker": "ecclesiastes, qoheleth, kohelet, coheleth",
    "Hooglied": "song of songs, song of solomon, canticum canticorum, canticum, cantica, canticles",
    "Jesaja": "isaiah, isaias, esaias, iesaias", "Jeremia": "jeremiah, jeremias, ieremias",
    "Klaagliederen": "lamentations, threni, lamentationes", "Ezechiel": "ezekiel", "Daniel": "",
    "Hosea": "osee, hoseas", "Joel": "ioel", "Amos": "", "Obadja": "obadiah, abdias, obadias", "Jona": "jonah, jonas, ionas",
    "Micha": "micah, michaeas, micheas", "Nahum": "", "Habakuk": "habakkuk, habacuc", "Sefanja": "zephaniah, sophonias, zefanja",
    "Haggai": "haggaeus, aggaeus", "Zacharia": "zechariah, zacharias", "Maleachi": "malachi, malachias",
    "Mattheus": "matthew, matteus, matthaeus", "Marcus": "mark, markus", "Lucas": "luke, lukas",
    "Johannes": "john, iohannes, ioannes, joannes", "Handelingen": "acts, acta, acta apostolorum",
    "Romeinen": "romans, ad romanos", "Galaten": "galatians, ad galatas", "Efeziers": "ephesians, ad ephesios",
    "Filippenzen": "philippians, ad philippenses", "Kolossenzen": "colossians, ad colossenses", "Titus": "ad titum",
    "Filemon": "philemon, ad philemonem", "Hebreeen": "hebrews, ad hebraeos", "Jakobus": "james, jacobus, iacobus, iacobi",
    "Judas": "jude, iudae", "Openbaring": "revelation, apocalypse, apocalypsis, apocalyps",
}
# the numbered books: their names without the number, and the Dutch name of the family
_FAMILIES = {
    "Samuel": "samuelis", "Koningen": "kings, regum", "Kronieken": "chronicles, chronicorum, paralipomenon",
    "Korintiers": "corinthians, ad corinthios", "Tessalonicenzen": "thessalonians, ad thessalonicenses",
    "Timotheus": "timothy, ad timotheum", "Petrus": "peter, petri", "Johannes": "john, iohannis, ioannis",
}
_ORDINAL = {"1": 1, "i": 1, "first": 1, "eerste": 1, "2": 2, "ii": 2, "second": 2, "tweede": 2, "3": 3, "iii": 3,
            "third": 3, "derde": 3}


def _fold(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace(".", " ").split())


def _phrases(names: str) -> list[str]:
    return [_fold(n) for n in names.split(",") if n.strip()]


def _names() -> dict:
    return {n: book for book, names in _ALIASES.items() for n in [_fold(book)] + _phrases(names)}


def canonical(name: str) -> str | None:
    """One of the 66 BOOKS for a book as a model or a reader writes it ("Ecclesiastes", "1 Corinthians",
    "I Samuelis", "Hebreeën"), or None when it is none of them (an apocryphal book, a title)."""
    s = _fold(name or "")
    if not s:
        return None
    if name in CHAPTERS:
        return name
    m = s.split(" ", 1)
    if len(m) == 2 and m[0] in _ORDINAL:
        n, rest = _ORDINAL[m[0]], m[1]
        for family, names in _FAMILIES.items():
            if rest == _fold(family) or rest in _phrases(names):
                book = f"{n} {family}"
                return book if book in CHAPTERS else None
        return None
    return _NAMES.get(s)


_NAMES = _names()


# ---------------------------------------------------------------- a title that treats many books

MANY_RX = re.compile(r"\b(?:omnes|universas|vniuersas)\b[^.:;]{0,30}\b(?:epistolas|prophetas|libros)\b|"
                     r"\bepistolas\s+(?:canonicas|catholicas)|"
                     r"\b(?:duodecim|xii\.?)\s+prophet|\bprophet(?:as|arum)\s+(?:minor|maior)|"
                     r"\bquat(?:u)?or\s+evangel|\bevangelistas\b|\bharmonia\w*\s+(?:quat|evangel)|"
                     r"\b(?:novum|nouum|vetus)\s+testamentum\b|\btestamentum\s+(?:novum|nouum|vetus)\b|"
                     r"\bveteris\s+testamenti\s+libros|\bbiblia\s+sacra|^biblia\b|"
                     r"\bpentateuch|\b(?:quinque\s+)?libros\s+mosis|\bmosis\s+libri\s+(?:v|quinque)\b|\bepistolas\s+canonicas", re.I)
TWO_RX = re.compile(r"\b(?:utramq\w*|vtramq\w*|ambas|utriusq\w*|vtriusq\w*)\b[^.:;]{0,40}\bepistol", re.I)


def many_books(title: str) -> int:
    """How many Bible books a title says the work treats at least: 3 for "in omnes epistolas", "Novum Testamentum",
    "prophetae minores", "quinque libros Mosis"; 2 for "utramque epistolam"; else 0. Not the genitive of one
    letter ("Epistolae Pauli ad Galatas") or "Opera omnia"."""
    return 3 if MANY_RX.search(title) else 2 if TWO_RX.search(title) else 0
