"""All analyses of a Latin word from Whitaker's Words, each as lemma, code, parse and senses.

The code follows the morphology codes of the Logos Latin interlinear: VRAI3P is a verb, perfect,
active, indicative, 3rd person, plural; NCASF a common noun, accusative, singular, feminine.
`describe` spells a code out in words."""

TENSE = {"P": "present", "I": "imperfect", "F": "future", "R": "perfect", "L": "pluperfect", "T": "future perfect"}
VOICE = {"A": "active", "P": "passive", "D": "deponent"}
MOOD = {"I": "indicative", "S": "subjunctive", "M": "imperative", "N": "infinitive", "P": "participle",
        "G": "gerund", "D": "gerundive", "U": "supine"}
CASE = {"N": "nominative", "G": "genitive", "D": "dative", "A": "accusative", "B": "ablative",
        "V": "vocative", "L": "locative"}
NUMBER = {"S": "singular", "P": "plural"}
GENDER = {"M": "masculine", "F": "feminine", "N": "neuter", "C": "common"}
PRON = {"P": "personal", "D": "demonstrative", "R": "relative", "I": "interrogative/indefinite",
        "X": "reflexive", "A": "pronominal adjective"}
DEGREE = {"C": "comparative", "S": "superlative"}
SIMPLE = {"B": "adverb", "C": "conjunction", "P": "preposition", "I": "interjection", "M": "numeral",
          "X": "other"}


def _cng(s):
    out = [CASE.get(s[0:1]), NUMBER.get(s[1:2]), GENDER.get(s[2:3])]
    return [x for x in out if x]


def describe(code: str) -> str:
    """VRAI3P -> 'verb, perfect active indicative, 3rd person plural'."""
    if not code:
        return ""
    head, rest = code[0], code[1:]
    if head == "V":
        t, v, m, tail = rest[0:1], rest[1:2], rest[2:3], rest[3:]
        words = ["verb,", TENSE.get(t, ""), VOICE.get(v, ""), MOOD.get(m, "")]
        if tail.startswith("-"):
            words += _cng(tail[1:])
        elif tail:
            p = {"1": "1st", "2": "2nd", "3": "3rd"}.get(tail[0:1], "")
            words += [f"{p} person" if p else "", NUMBER.get(tail[1:2], "")]
        return " ".join(w for w in words if w)
    if head == "N":
        kind = {"C": "noun", "P": "proper noun"}.get(rest[0:1], "noun")
        return ", ".join([kind] + [" ".join(_cng(rest[1:]))])
    if head == "J":
        deg = DEGREE.get(rest[3:4], "")
        return "adjective, " + " ".join(_cng(rest[0:3]) + ([deg] if deg else []))
    if head == "R":
        kind = PRON.get(rest[0:1], "")
        p = {"1": "1st person", "2": "2nd person", "3": "3rd person"}.get(rest[1:2], "")
        return f"{kind} pronoun, " + " ".join(([p] if p else []) + _cng(rest[2:]))
    return SIMPLE.get(head, code)


# ------------------------------------------------------------ Whitaker's Words

W_TENSE = {"PRES": "P", "IMPF": "I", "FUT": "F", "PERF": "R", "PLUP": "L", "FUTP": "T"}
W_VOICE = {"ACTIVE": "A", "PASSIVE": "P"}
W_MOOD = {"IND": "I", "SUB": "S", "IMP": "M", "INF": "N", "PPL": "P"}
W_CASE = {"NOM": "N", "GEN": "G", "DAT": "D", "ACC": "A", "ABL": "B", "VOC": "V", "LOC": "L"}
W_PRON = {"PERS": "P", "DEMONS": "D", "REL": "R", "INTERR": "I", "INDEF": "I", "REFLEX": "X", "ADJECT": "A"}
W_DEGREE = {"COMP": "C", "SUPER": "S"}
# Forms Whitaker lists as "uniques" without a dictionary form.
UNIQUE_LEMMA = {"sum": "sum", "es": "sum", "est": "sum", "sumus": "sum", "estis": "sum", "sunt": "sum",
                "esse": "sum", "fore": "sum", "eram": "sum", "erat": "sum", "erant": "sum", "esset": "sum",
                "essent": "sum", "sit": "sum", "sint": "sum", "fuit": "sum", "fuerunt": "sum",
                "ero": "sum", "erit": "sum", "erunt": "sum", "possum": "possum", "potest": "possum",
                "posse": "possum", "possit": "possum", "possint": "possum", "possunt": "possum",
                "poterat": "possum", "potuisset": "possum"}
# The rest of Whitaker's irregular forms by the start of their meaning: fuerit, simus, vult, mavult, memento
UNIQUE_SENSES = [("to be, exist", "sum"), ("be;", "sum"), ("be away/absent", "absum"), ("come hither", "adsum"),
                 ("it is necessary", "necesse est"), ("let them be treated", "ago"), ("let it be treated", "ago"),
                 ("give/bring here", "cedo"), ("prefer", "malo"), ("remember", "memini"), ("be willing", "volo"),
                 ("be unwilling", "nolo"), ("do you not see", "video"), ("god", "deus"), ("ox", "bos"),
                 ("same", "idem")]

# Forms this version of Whitaker's Words lacks altogether: the infinitives, imperatives, forem and futurus of sum,
# and necesse. Added as candidates of their own, in the same shape as Whitaker's.
_BE = "to be, exist; also used to form verb perfect passive tenses with NOM PERF PPL"
EXTRA = {
    "esse": [("sum", "VPAN", _BE)], "fuisse": [("sum", "VRAN", _BE)], "fore": [("sum", "VFAN", _BE)],
    "es": [("sum", "VPAM2S", _BE)], "este": [("sum", "VPAM2P", _BE)], "esto": [("sum", "VFAM2S", _BE), ("sum", "VFAM3S", _BE)],
    "estote": [("sum", "VFAM2P", _BE)], "sunto": [("sum", "VFAM3P", _BE)],
    "forem": [("sum", "VIAS1S", _BE)], "fores": [("sum", "VIAS2S", _BE)], "foret": [("sum", "VIAS3S", _BE)],
    "foremus": [("sum", "VIAS1P", _BE)], "foretis": [("sum", "VIAS2P", _BE)], "forent": [("sum", "VIAS3P", _BE)],
    "futurus": [("sum", "VFAP-NSM", _BE)], "futura": [("sum", "VFAP-NSF", _BE), ("sum", "VFAP-NPN", _BE), ("sum", "VFAP-APN", _BE)],
    "futurum": [("sum", "VFAP-NSN", _BE), ("sum", "VFAP-ASN", _BE), ("sum", "VFAP-ASM", _BE)],
    "futuri": [("sum", "VFAP-NPM", _BE), ("sum", "VFAP-GSM", _BE), ("sum", "VFAP-GSN", _BE)],
    "futurae": [("sum", "VFAP-GSF", _BE), ("sum", "VFAP-DSF", _BE), ("sum", "VFAP-NPF", _BE)],
    "necesse": [("necesse", "J---", "necessary, unavoidable, inevitable; (necesse est: it is necessary)")],
}
# Spellings of the time Whitaker does not know, tried when a word gets no analysis as it stands:
# shortened perfects (negarunt, probasse) and an m before t without the p (assumtio, redemtio)
ALTERNATES = [(r"arunt$", "averunt"), (r"erunt$", "everunt"), (r"asse$", "avisse"), (r"assem$", "avissem"),
              (r"asset$", "avisset"), (r"assent$", "avissent"), (r"(?<=[aeiou])mt", "mpt"), (r"mpt", "mt")]


def latin_spelling(s: str) -> str:
    """Whitaker writes u for v: a u at the start or between vowels before a vowel is a v."""
    out = []
    for i, c in enumerate(s):
        nxt = s[i + 1:i + 2]
        prev = s[i - 1:i] if i else ""
        if c == "u" and nxt in set("aeiou") and (not prev or (prev in "aeio" and nxt != "u")):
            c = "v"
        out.append(c)
    return "".join(out)


def _f(infl, name):
    v = infl.features.get(name)
    return v.name if v is not None else "X"


def _cng_code(infl):
    return (W_CASE.get(_f(infl, "Case"), "-") + {"S": "S", "P": "P"}.get(_f(infl, "Number"), "-")
            + {"M": "M", "F": "F", "N": "N", "C": "C"}.get(_f(infl, "Gender"), "-"))


def code_of(infl, lexeme) -> str:
    from whitakers_words.enums import WordType as T
    wt = infl.wordType
    deponent = "DEP" in (getattr(lexeme, "form", None) or [])
    if wt in (T.V, T.VPAR):
        tense = W_TENSE.get(_f(infl, "Tense"), "-")
        voice = "D" if deponent else W_VOICE.get(_f(infl, "Voice"), "-")
        if wt == T.VPAR:
            return f"V{tense}{voice}P-{_cng_code(infl)}"
        mood = W_MOOD.get(_f(infl, "Mood"), "-")
        if mood == "N":
            return f"V{tense}{voice}N"
        person = _f(infl, "Person")
        return f"V{tense}{voice}{mood}{person if person in '123' else '-'}{_f(infl, 'Number').replace('X', '-')}"
    if wt == T.N:
        proper = "P" if (getattr(lexeme, "form", None) or ["", ""])[1:2] == ["N"] else "C"
        return f"N{proper}{_cng_code(infl)}"
    if wt == T.ADJ:
        return f"J{_cng_code(infl)}{W_DEGREE.get(_f(infl, 'Degree'), '')}"
    if wt == T.PRON:
        kind = W_PRON.get((getattr(lexeme, "form", None) or ["X"])[0], "") or W_PRON.get(_f(infl, "PronounType"), "-")
        return f"R{kind}-{_cng_code(infl)}"
    if wt == T.NUM:
        return f"M{_cng_code(infl)}"
    if wt == T.ADV:
        return "B" + W_DEGREE.get(_f(infl, "Degree"), "")
    return {T.CONJ: "C", T.PREP: "P", T.INTERJ: "I"}.get(wt, "X")


PRONOUNS = {(3, 1): "hic", (4, 1): "is", (4, 2): "idem", (5, 1): "ego", (5, 2): "tu", (5, 3): "nos",
            (5, 4): "sui", (5, 5): "vos", (6, 2): "ipse"}


def pronoun_lemma(roots, cat, kind) -> str:
    """Whitaker's pronoun stems are bare (qu, h, ill): the nominative by declension."""
    key = tuple(cat[:2])
    if key in PRONOUNS:
        return PRONOUNS[key]
    if cat[:1] == [1]:  # qui / quis and compounds (aliquis)
        return roots[0] + ("i" if kind[0] in ("REL", "ADJECT") else "is")
    if cat[:1] == [6]:  # ille, iste
        return roots[0] + "e"
    return roots[0]


def lemma_of(analysis, form_text: str) -> str:
    """The dictionary form: first principal part, nominative singular (masculine)."""
    from whitakers_words.enums import WordType as T
    from whitakers_words.finder import find_infl
    lx = analysis.lexeme
    roots = list(lx.roots or [])
    try:
        if not roots:   # one of Whitaker's irregular forms: no stems, only its meaning tells the word
            if form_text in UNIQUE_LEMMA:
                return UNIQUE_LEMMA[form_text]
            senses = " ".join(lx.senses or [])
            return next((lem for start, lem in UNIQUE_SENSES if senses.startswith(start)), form_text)
        cat = lx.category
        if lx.wordType == T.V:
            if roots[0] in ("s", "es") or cat[:1] == [5]:  # sum and its compounds: possum, prosum
                return "sum" if roots[0] in ("s", "es", "fu") else roots[0].removesuffix("s") + "sum"
            try:
                lem = roots[0] + find_infl(T.V, cat, ["PRES", "ACTIVE", "IND", "1", "S"])
            except Exception:   # irregular (edo): the first person as most verbs have it
                lem = roots[0] + "o"
            return lem + "r" if "DEP" in (getattr(lx, "form", None) or []) and lem.endswith("o") else lem
        if lx.wordType == T.N:
            return roots[0] + find_infl(T.N, cat, ["NOM", "S", lx.form[0]])
        if lx.wordType == T.ADJ:
            return roots[0] + find_infl(T.ADJ, cat, ["NOM", "S", "M", "POS"])
        if lx.wordType == T.PRON:
            return pronoun_lemma(roots, cat, getattr(lx, "form", None) or [""])
        if lx.wordType == T.NUM:
            return roots[0] + find_infl(T.NUM, cat, ["NOM", "S", "M"])
    except Exception:
        pass
    return roots[0]


def gerund_split(code: str) -> list[str]:
    """Whitaker gives the gerundive as a future passive participle and has no gerund: a gerundive
    V-PD-<case number gender>, and for a neuter singular outside the nominative also a gerund V-AG-<case>."""
    if not code.startswith("VFPP-"):
        return [code]
    cng = code[5:]
    out = [f"V-PD-{cng}"]
    if cng[1:3] == "SN" and cng[0] in "GDAB":
        out.append(f"V-AG-{cng[0]}")
    if cng[1:3] == "SM" and cng[0] in "GDB":  # Whitaker often lists the masculine only
        out.append(f"V-AG-{cng[0]}")
    return out


class Analyzer:
    def __init__(self):
        from whitakers_words.parser import Parser
        # All ages (the texts are 16th-17th century) and all but the rarest words; Whitaker's
        # default is classical Latin only, which misses intelligo, existo, magis, suus.
        self.parser = Parser(age="H", frequency="F")
        self.memo = {}

    def analyse(self, norm: str) -> list[dict]:
        """All distinct (lemma, code) analyses of a normalized word; [] if Whitaker has none. With the
        forms Whitaker lacks (EXTRA), and a spelling of the time read as Whitaker's (ALTERNATES)."""
        if norm in self.memo:
            return self.memo[norm]
        out = self.whitaker(norm)
        for lemma, code, senses in EXTRA.get(norm, []):
            out.append({"lemma": lemma, "code": code, "parse": describe(code), "senses": senses})
        if not out:
            import re
            for pat, rep in ALTERNATES:
                alt = re.sub(pat, rep, norm)
                if alt != norm and (found := self.whitaker(alt)):
                    out = found
                    break
        out.sort(key=lambda c: (c["lemma"], c["code"], c["senses"]))   # a fixed order: Whitaker's own follows Python's set order, which changes per run
        self.memo[norm] = out
        return out

    def whitaker(self, norm: str) -> list[dict]:
        out, seen = [], set()
        try:
            word = self.parser.parse(norm)
        except Exception:
            word = None
        for form in (word.forms if word else []):
            for a in form.analyses.values():
                lemma = latin_spelling(lemma_of(a, form.text))
                enc = form.enclitic.text if form.enclitic else ""
                senses = "; ".join(a.lexeme.senses or [])
                for infl in a.inflections:
                    for code in gerund_split(code_of(infl, a.lexeme)):
                        key = (lemma, code, enc)
                        if key in seen:
                            continue
                        seen.add(key)
                        parse = describe(code) + (f" + enclitic -{enc}" if enc else "")
                        out.append({"lemma": lemma + (f" + -{enc}" if enc else ""), "code": code,
                                    "parse": parse, "senses": senses})
        return out
