"""Is this a Latin word? Whitaker's Words (whitakers_words, requirements.txt) with every inflection.

known() looks a word up as an early modern print spells it: ſ as s, æ/œ as ae/oe, accents and the macron of an
abbreviation dropped, u/v and i/j as Whitaker takes them anyway. Its sqlite database is read, not loaded whole (the
module's default import holds the whole dictionary in memory, too much on the server). Every frequency and every age is
admitted: a rare or late word is a word. A few common words Whitaker lacks are added (EXTRA).

Used by transcription/latin_words.py; tested in tests/test_latin_words.py. Over 39 Latin works (13 November 2026) it
knew 89% of the words; most of the rest are names, abbreviations (lib., tit.) and words of the trades.
"""
import functools
import re
import unicodedata

EXTRA = {"fuisse", "fuere", "necesse", "foret", "forent", "fieret", "fierent", "quot", "totidem", "sese"}
_parser = None


def norm(word: str) -> str:
    """The spelling Whitaker looks up: lower case, no ligatures, no long s, no accents."""
    w = word.replace("æ", "ae").replace("Æ", "Ae").replace("œ", "oe").replace("Œ", "Oe").replace("ſ", "s")
    w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
    return w.lower()


@functools.lru_cache(maxsize=500_000)
def _known(w: str) -> bool:
    global _parser
    if w in EXTRA:
        return True
    if not re.fullmatch(r"[a-z]+", w):
        return False
    if _parser is None:
        from whitakers_words.parser import Parser
        _parser = Parser(database=True, frequency="Z", age="H")
    try:
        return len(_parser.parse(w).forms) > 0
    except Exception:   # the parser raises on what it cannot take apart
        return False


def known(word: str) -> bool:
    """Is word (one word, as printed) a form of a Latin word?"""
    return _known(norm(word))
