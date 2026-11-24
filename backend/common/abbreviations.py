"""The printers' signs of abbreviation and the letters they stand for: q; for -que, a stroke or tilde over a vowel
for m or n (cū cum, tātum tantum, nō non), ꝑ per, ꝓ pro, ꝗ and ꝙ quod, ꝰ -us, ꝯ con, a few whole words (oēs omnes,
Dñs Dominus). The transcription keeps the sign as printed (as CATMuS and TEI's <am>/<ex> do); the reader shows the
word written out with the printed sign on hover, the search finds it written out, the TEI download keeps both.

Made on the words that occur most in the transcriptions (a sample of 500 book.xml, 13 November 2026). Some signs mean
more than one thing; where a sign stands alone the commonest reading is taken: ꝗ and ꝙ quod, q̃ quae, ē and ō the
letter (mostly è and ò read as a mark), ȳ the verse sign (v, see CLAUDE.md), ꝯ cum, ꝫ et. Greek, a word with a
digit, a Spanish ñ and the printers' accents (quàm) are not abbreviations and stay.
"""
import re
import unicodedata

MACRON, TILDE = "̄", "̃"
NASAL = {"ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ã": "a", "ẽ": "e", "ĩ": "i", "õ": "o", "ũ": "u",
         "Ā": "A", "Ē": "E", "Ī": "I", "Ō": "O", "Ū": "U", "Ã": "A", "Ẽ": "E", "Ĩ": "I", "Õ": "O", "Ũ": "U"}
# whole words, in lower case; a capital first letter is kept
WORDS = {"oēs": "omnes", "oē": "omne", "oēm": "omnem", "oīa": "omnia", "oībus": "omnibus", "oīs": "omnis",
         "oīum": "omnium", "oīno": "omnino", "dñs": "dominus", "dñi": "domini", "dño": "domino", "dñm": "dominum",
         "dñe": "domine", "qñ": "quando", "nō": "non", "nõ": "non", "quōd": "quod", "q̃d": "quod", "qd̃": "quod",
         "ꝗd": "quod", "q̃modo": "quomodo", "q̄modo": "quomodo", "q̃": "quae", "q̄": "quae", "ꝗ": "quod", "ꝙ": "quod",
         "ꝑ": "per", "ꝓ": "pro", "ꝯ": "cum", "ꝫ": "et", "ē": "e", "ẽ": "e", "ō": "o", "õ": "o", "ā": "a", "ã": "a",
         "ī": "i", "ĩ": "i"}
GERMAN_WORDS = {"vñ": "vnd", "uñ": "und"}
QUE = ("q;", "q̃", "q̄", "qꝫ", "cꝫ", "qꝙ", "qꝗ", "ꝫ", "ꝙ", "ꝗ")   # at the end of a word after other letters: -que
SIGN = {"ꝓ": "pro", "ꝰ": "us", "Ꝓ": "Pro"}
WORD = re.compile(r"[\ẁ-ͯ]+")
FOREIGN = re.compile(r"[Ͱ-Ͽἀ-῿֐-׿\d]")
SIGNS = set("ꝑꝓꝗꝙꝯꝫꝰȳñŇ" + "".join(NASAL) + "ꝐꝒꝖꝘꝮ") | {MACRON, TILDE}
VOWELS = set("aeiouyæœ")


def _case(printed: str, full: str) -> str:
    return full[0].upper() + full[1:] if printed[:1].isupper() and full else full


def _after(text: str, end: int) -> str:
    """The letter after a word broken at the line end (cō-|modè), else ""."""
    m = re.match(r"[-¬⸗­=]\s*([^\W\d_])", text[end:])
    return m.group(1).lower() if m else ""


def _nasal(vowel: str, nxt: str, german: bool, con: bool = False) -> str:
    """A vowel with a stroke: the vowel and m or n. con: the word opens with co (cōnecto, cōuenit: con-)."""
    if vowel.lower() == "i":
        return vowel   # hīc, hominī: a long i, no letter left out
    if not nxt:
        return vowel + ("n" if german or vowel.lower() == "o" else "m")   # cū cum, autē autem; nō non; geborē geboren, tamē tamen
    if nxt in VOWELS - {"u"} or nxt in "rlh":
        return vowel   # periērat: an accent read as a stroke; no n or m before a vowel or r
    if nxt in "nuv" and not german:
        return vowel + ("n" if con else "m")   # dāno damno, Quāuis Quamuis; cōnecto, cōuenit
    return vowel + ("m" if nxt in "bpm" else "n")


def _in_word(w: str, start: int, text: str, german: bool) -> list:
    """The signs inside one word: [(start, end, written out)]."""
    out, i, n = [], 0, len(w)
    base = [unicodedata.normalize("NFD", c)[0].lower() for c in w]
    while i < n:
        c = w[i]
        nxt_i = i + 1
        if nxt_i < n and w[nxt_i] in (MACRON, TILDE):   # a mark written as a combining letter (cū)
            c, nxt_i = w[i:nxt_i + 1], nxt_i + 1
        rest = w[i:]
        # -que at the end: atq; neq̃ atqꝫ eodemꝗ semperꝙ
        if i and any(rest == q for q in QUE):
            out.append((start + i, start + n, "que"))
            break
        nxt = base[nxt_i] if nxt_i < n else _after(text, start + n)
        if c in ("ꝑ", "Ꝑ"):
            out.append((start + i, start + i + 1, _case(c, "pro" if nxt in ("p", "b") else "per")))   # ꝑpheta: ꝓ read as ꝑ
        elif c in SIGN:
            out.append((start + i, start + i + 1, SIGN[c]))
        elif c in ("ꝯ", "Ꝯ"):
            out.append((start + i, start + i + 1, _case(c, "com" if nxt in ("b", "p", "m") else "con")))
        elif c == "ȳ":
            out.append((start + i, start + i + 1, "j" if i and base[i - 1] == "i" else "ij"))   # subȳcere, aliȳs alijs
        elif c == "ñ" and i and (german or nxt not in VOWELS):
            out.append((start + i, start + i + 1, "nn"))   # weñ wenn, erkeñen erkennen; Señor stays
        elif c in NASAL or (len(c) == 2 and c[0].lower() in "aeiou"):
            v = NASAL.get(c) or c[0]
            prev = base[i - 1] if i else ""
            out.append((start + i, start + nxt_i, _nasal(v, nxt, german or (prev == "m" and v.lower() == "e"),   # tamē nomē
                                                         con=i == 1 and base[0] == "c" and v.lower() == "o")))
        # a stroke over a consonant (p̄ pre/pro/præ, t̄ tur/tus) means too many things: it stays as printed
        i = nxt_i
    return out


def abbreviations(text: str, german: bool = False) -> list:
    """[(start, end, written out)] for every sign of abbreviation in text; text[start:end] is what was printed.
    german: the text is German (a nasal stroke at the end is -n, vñ is vnd)."""
    out = []
    for m in WORD.finditer(text):
        a, b = m.span()
        w = m.group(0)
        if b < len(text) and text[b] == ";" and w.endswith("q"):
            w, b = w + ";", b + 1
        if FOREIGN.search(w) or not (set(w) & SIGNS or w.endswith("q;")):
            continue
        low = w.lower()
        if w == "q;":   # q; alone: the -que of the word before (Rogat q; iterum)
            if a >= 2 and text[a - 1] == " " and text[a - 2].isalpha():
                out.append((a - 1, b, "que"))
            continue
        if w == "ȳ":   # the verse sign
            out.append((a, b, "v"))
            continue
        whole = (GERMAN_WORDS if german else {}).get(low) or WORDS.get(low)
        if whole and not (low in ("ē", "ō", "ā", "ã", "ẽ", "õ") and _after(text, b)):   # cō- ō-: a broken word
            out.append((a, b, _case(w, whole)))
            continue
        out += _in_word(w, a, text, german)
    return out


def expand(text: str, german: bool = False) -> str:
    """text with every sign of abbreviation written out."""
    parts, at = [], 0
    for a, b, full in abbreviations(text, german):
        parts += [text[at:a], full]
        at = b
    return "".join(parts) + text[at:]
