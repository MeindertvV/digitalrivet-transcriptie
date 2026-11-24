"""Whether an answer of the model is a translation that may be kept (REFACTOR_PLAN.md step 78: was the checking
part of translate.py, which still has the names).

parse_answer reads the units out of an answer and says of each one why it cannot be kept: written in another
script (foreign_script), left in Latin (copied, partly_copied; not for a text that is not Latin at all, modern),
much shorter or longer than its Latin, or with the model's own comment in it. tidy keeps the italics of the
transcription and nothing else.
"""

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.units import TAG, Unit, plain  # noqa: E402

# GLM sometimes ends its answer with its own tool-call markup (</arg_value></tool_call>) and noise after it:
# the translation is what comes before the first such tag.
LEAK = re.compile(r"</?(?:arg_value|arg_key|tool_call)>")


# The model talking about its own answer ("Let me write the translation properly.", "... no."):
# anything like it on a line of its own, and inside a unit only what no author writes
# ("Let me at least offer fine thread" is Pareus).
META = re.compile(r"\b(let me|i will now|i'll|here is the translation)\b|\.\.\.\s*no\b", re.I)
META_IN_UNIT = re.compile(r"\blet me (write|redo|try|translate|re-?translate|fix|correct|check|revise)\b"
                          r"|\bhere is the translation\b|\.\.\.\s*no\b|\bwait\s*[—–:,-]|\bhmm\b|\bnext message\b"
                          r"|\bmust translate\b|\btranslate (it )?as\b|(?-i:\bDONE\b)", re.I)
UNIT_LINE = re.compile(r"\s*\[\d+\]")
# A remark the model adds after its last unit ("Note: I have kept ...", "Translator's note: ..."): no part of the unit.
NOTE = re.compile(r"\W*(?:translator\W?s?\W?\s*|vertalers?\s*)?(?:notes?\b|n\.\s?b\.?|nb\b|opmerking(?:en)?\b|noot\b)\W{0,3}:", re.I)
ABOUT_ITSELF = re.compile(r"\bI\b|\b[Ii]k\b|\b[Tt]ranslat|\b[Vv]erta")
NOTA = re.compile(r"\bnota|\bN\.\s?B\b", re.I)

# A translation is written in Latin script. Greek and Hebrew are allowed because the prompt
# keeps the original of a quotation in square brackets after its translation; Chinese,
# Japanese, Korean, Indic and every other script mean the model broke down or answered in
# the wrong language, and such a unit must never be kept.
ALLOWED_RANGES = (
    (0x0041, 0x005A), (0x0061, 0x007A), (0x00AA, 0x00AA), (0x00B5, 0x00B5),
    (0x00BA, 0x00BA), (0x00C0, 0x00D6), (0x00D8, 0x00F6), (0x00F8, 0x02FF),
    (0x0370, 0x03FF), (0x0590, 0x05FF), (0x1D00, 0x1DBF), (0x1E00, 0x1EFF),
    (0x1F00, 0x1FFF), (0x2070, 0x209F), (0x2100, 0x214F), (0x2C60, 0x2C7F),
    (0xA720, 0xA7FF), (0xAB30, 0xAB6F), (0xFB00, 0xFB06), (0xFB1D, 0xFB4F),
    (0x10780, 0x107BF), (0x1D400, 0x1D7FF), (0x1DF00, 0x1DFFF),
)
FOREIGN_SCRIPTS = (
    ("Chinese, Japanese or Korean characters", (
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
        (0x20000, 0x2FA1F), (0x30000, 0x323AF),
    )),
    ("Japanese kana", ((0x3040, 0x30FF), (0x31F0, 0x31FF))),
    ("Korean Hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7FF))),
    ("Indic characters", (
        (0x0900, 0x0DFF), (0x1CD0, 0x1CFF), (0xA8E0, 0xA8FF), (0xA980, 0xA9DF),
        (0x11000, 0x1107F),
    )),
    ("Thai characters", ((0x0E00, 0x0E7F),)),
    ("Lao characters", ((0x0E80, 0x0EFF),)),
    ("Tibetan characters", ((0x0F00, 0x0FFF),)),
    ("Burmese characters", ((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F))),
    ("Khmer characters", ((0x1780, 0x17FF), (0x19E0, 0x19FF))),
    ("Georgian characters", ((0x10A0, 0x10FF), (0x1C90, 0x1CBF))),
    ("Ethiopic characters", ((0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F))),
    ("Arabic characters", (
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFC),
    )),
    ("Syriac characters", ((0x0700, 0x074F), (0x0860, 0x086F))),
    ("Armenian characters", ((0x0530, 0x058F),)),
    ("Cyrillic characters", (
        (0x0400, 0x052F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F),
    )),
    ("CJK characters or punctuation", (
        (0x2E80, 0x2EFF), (0x2F00, 0x2FDF), (0x3000, 0x303F), (0x3100, 0x312F),
        (0x31A0, 0x31EF), (0x3200, 0x33FF), (0xFE10, 0xFE1F), (0xFE30, 0xFE4F),
        (0xFF00, 0xFFEF),
    )),
    ("Braille patterns", ((0x2800, 0x28FF),)),
    ("emoji or pictographs", ((0x1F000, 0x1FAFF),)),
)


def in_ranges(cp: int, ranges) -> bool:
    for start, end in ranges:
        if start <= cp <= end:
            return True
    return False


def char_name(ch: str) -> str:
    """A character as ASCII text for a rejection message (the console need not be UTF-8)."""
    return f"U+{ord(ch):04X} {unicodedata.name(ch, 'unnamed')}"


def foreign_script(text: str) -> str | None:
    """Why `text` is not in a script a translation may use, or None.

    Latin is the script of a translation; Greek and Hebrew are allowed because the prompt
    keeps the original of a quoted text in square brackets after the translation. Chinese,
    Japanese, Korean, Indic and every other script mean the model broke down, so the unit
    is rejected and asked again instead of being stored."""
    for ch in text:
        cp = ord(ch)
        if cp < 0x80 or in_ranges(cp, ALLOWED_RANGES):
            continue
        for label, ranges in FOREIGN_SCRIPTS:
            if in_ranges(cp, ranges):
                return f"contains {label} ({char_name(ch)})"
        if unicodedata.category(ch).startswith("M"):  # a mark of an allowed script
            continue
        if unicodedata.category(ch).startswith("L"):
            return f"contains a non-Latin script ({char_name(ch)})"
        if cp == 0xFFFD:
            return "contains a replacement character"
        if ch.isspace():
            continue
    return None


# Problems that may mean a unit's text went under another number: its neighbours are not trusted either.
SHIFTED = ("missing", "much shorter than the Latin", "much longer than the Latin")


def without_note(text: str, units: list[Unit]) -> tuple[str, str | None]:
    """The answer without a remark of the model after its last unit, and that remark (None if there is none). A
    remark begins with "Note:" or the like, after an empty line or speaking of the model or its translation; where the
    Latin of that unit has a nota of its own, the line is the unit's and stays."""
    lines = text.splitlines()
    numbered = [i for i, line in enumerate(lines) if UNIT_LINE.match(line)]
    if not numbered:
        return text, None
    k = int(re.match(r"\s*\[(\d+)\]", lines[numbered[-1]]).group(1))
    if 1 <= k <= len(units) and NOTA.search(plain(units[k - 1].la)):
        return text, None
    for i in range(numbered[-1] + 1, len(lines)):
        if NOTE.match(lines[i]) and (not lines[i - 1].strip() or ABOUT_ITSELF.search(lines[i])):
            return "\n".join(lines[:i]), " ".join(line.strip() for line in lines[i:] if line.strip())
    return text, None


def parse_answer(text: str, units: list[Unit]) -> tuple[dict[int, str], dict[int, str], str | None]:
    """The translations in an answer: ({k: translation} of the usable units, {k: reason} of the
    others, a reason the whole answer is unusable or None); k counts the units from 1."""
    if len(units) == 1 and len(re.findall(r"\[1\]", text)) > 1:
        # One title line asked, several drafts given (GLM): the last one that says something.
        drafts = [re.sub(r"\s*##\s*$", "", d).strip() for d in re.split(r"\[1\]", text)[1:]]
        text = "[1] " + next((d for d in reversed(drafts) if d.strip("# ")), "")
    out: dict[int, str] = {}
    last = None
    answer, note = without_note(text, units)
    if note:
        print(f"    note: a remark of the model after the last unit left out ({note[:60]!r})")
    for line in answer.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s?(.*)", line)
        if m:
            last = int(m.group(1))
            if last in out:
                return {}, {}, f"unit [{last}] appears twice"
            out[last] = m.group(2).strip()
        elif last is not None and line.strip():
            out[last] += " " + line.strip()  # unit continued on a new line
    # A title page split over lines: a fragment of a few words ("illustrata.") may be taken
    # into the translation of the line before it and stay empty itself.
    for k in range(2, len(units) + 1):
        if not out.get(k) and out.get(k - 1) and len(plain(units[k - 1].la).split()) <= 3:
            out[k] = ""
    extra = [k for k in out if not 1 <= k <= len(units)]
    if extra:
        return {}, {}, f"extra unit(s) {extra[:10]}"
    if m := META.search("\n".join(l for l in text.splitlines() if not UNIT_LINE.match(l))):
        return {}, {}, f"contains a comment from the model ({m.group(0)!r})"
    bad = {k: "missing" for k in range(1, len(units) + 1) if k not in out}
    out = {k: re.sub(r"^#+\s*", "", tidy(v, units[k - 1].la)).strip() for k, v in out.items()}
    lost = [k for k in out if TAG.search(units[k - 1].la) and not TAG.search(out[k])]
    if lost:  # not worth failing a unit for: the words are there, only not in italics
        print(f"    note: unit(s) {lost[:10]} without the italics of the Latin")

    def latin_len(k):  # with the fragments after it that it took in
        n = len(plain(units[k - 1].la))
        while k < len(units) and out.get(k + 1) == "":
            k += 1
            n += len(plain(units[k - 1].la)) + 1
        return n
    for k, tr in out.items():
        la = plain(units[k - 1].la)
        if reason := foreign_script(tr):
            bad[k] = reason
        elif (copied(plain(tr), la) or partly_copied(plain(tr), la)) and not modern(units[k - 1].la):
            bad[k] = "left in Latin"
        # Much shorter than its Latin: it lost part of it, or its text went under another number.
        elif len(la) > 150 and len(plain(tr)) < 0.5 * len(la):
            bad[k] = "much shorter than the Latin"
        # Far longer: the model's thinking or several drafts ended up in the unit.
        elif len(plain(tr)) > 2.5 * latin_len(k) + 30:
            bad[k] = "much longer than the Latin"
        elif m := META_IN_UNIT.search(tr):
            bad[k] = f"contains a comment from the model ({m.group(0)!r})"
    for k in [k for k, r in bad.items() if r in SHIFTED]:
        for j in (k - 1, k + 1):
            if 1 <= j <= len(units) and j not in bad:
                bad[j] = "next to a unit that failed"
    for k in range(2, len(units) + 1):  # a fragment whose words went into a unit that failed
        if out.get(k) == "" and k - 1 in bad and k not in bad:
            bad[k] = "next to a unit that failed"
    return {k: v for k, v in out.items() if k not in bad}, bad, None


def tidy(translation: str, latin: str) -> str:
    """Keep the italics of the transcription and nothing else: drop Markdown *stars*, drop tags the
    Latin does not have, flatten nested tags and keep the words if the tags do not pair up."""
    translation = LEAK.split(translation)[0].rstrip()
    if "*" not in latin:  # Markdown italics: tags where the Latin has italics, else nothing
        translation = re.sub(r"\*\*?([^*]+?)\*\*?", r"<i>\1</i>" if TAG.search(latin) else r"\1", translation)
    if not TAG.search(latin):
        return plain(translation)
    out, stack = [], []
    for part in re.split(r"(</?(?:i|hi)>)", translation):
        m = re.fullmatch(r"<(/?)(i|hi)>", part)
        if not m:
            out.append(part)
        elif not m.group(1):
            if not stack:
                out.append(part)
            stack.append(m.group(2))
        elif stack:
            tag = stack.pop()
            if not stack:
                out.append(f"</{tag}>")
    if stack:  # an opening tag never closed
        return plain(translation)
    return re.sub(r"<(i|hi)>\s*</\1>", "", "".join(out))


# Source text that is not Latin at all (e.g. a modern notice): keeping it unchanged is not a failure.
MODERN = set("het een van dat niet met voor zijn wordt the and that which with this".split())


def modern(text: str) -> bool:
    words = re.findall(r"\w+", text.lower())
    return sum(w in MODERN for w in words) >= max(2, len(words) // 8)


def copied(translation: str, latin: str) -> bool:
    """True if most words of a translation are words of its Latin. Only lower-case
    words count: names such as "Ruben, Simeon, Levi" look the same in every language."""
    lower = re.compile(r"\b[a-zæœàâäéèêëïîôöùûüÿç]\w{3,}")
    src = set(lower.findall(latin))
    words = lower.findall(translation)
    return len(words) >= 5 and sum(w in src for w in words) > 0.5 * len(words)


def partly_copied(translation: str, latin: str, window: int = 15) -> bool:
    """True if a stretch of `window` words of the translation is nearly all words of its Latin:
    the model translated part of a long unit and left the rest in Latin."""
    lower = re.compile(r"\b[a-zæœàâäéèêëïîôöùûüÿç]\w{3,}")
    src = set(lower.findall(latin))
    hits = [w in src for w in lower.findall(translation)]
    if len(hits) < window:
        return False
    run = sum(hits[:window])
    best = run
    for i in range(window, len(hits)):
        run += hits[i] - hits[i - window]
        best = max(best, run)
    return best >= 0.9 * window
