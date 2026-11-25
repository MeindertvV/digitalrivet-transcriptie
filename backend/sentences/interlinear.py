"""An interlinear Latin-English edition of a stretch of a work.

For each unit (a thesis, a paragraph) of the work's translation file:

1. Sentences: the Latin and the English text are cut by sentences/split.py (. ? !, long ones at ; :),
   and Bertalign (LaBSE embeddings) pairs them, one or more Latin sentences with one or more English ones (a "bead").
2. Words: every Latin word gets all its analyses from Whitaker's Words (lemma, part of speech,
   form, English senses), each written as a short code (VRAI3P = verb, perfect, active,
   indicative, 3rd person, plural; see CODES below).
3. Choice: Haiku sees the Latin and English sentence and the candidates per word, and picks the
   analysis that fits the context, with a short gloss of the root ("to see"), a gloss of this
   form ("they saw") and the word(s) of the English sentence that render it. A word Whitaker does
   not know (a printing error, a Greek word, a name) gets its analysis from Haiku, marked as such.

Rows per word, top to bottom: Latin form, root (lemma), code, gloss of the root, gloss of the form,
English word. Above each bead its Latin sentence, below it its English one.

Scripture references in the margin (the <i>a Gen. 3. 6.</i> notes) and the note letters in the
text are kept apart: they are shown under the unit, not glossed.

Runs on the PC: LaBSE needs about 2 GB of memory, more than the server has free. A venv apart
from the pipeline's packages (the model itself, ~1.9 GB, comes on the first run). With torch for
CUDA LaBSE runs on the PC's GPU (Quadro T1000), many times faster than on the processor (about 100
paragraphs in 6 minutes there); cpu instead of cu128 in the index URL for a processor-only torch:
    python -m venv %USERPROFILE%\\.venvs\\interlinear-gpu
    %USERPROFILE%\\.venvs\\interlinear-gpu\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cu128
    %USERPROFILE%\\.venvs\\interlinear-gpu\\Scripts\\pip install sentence-transformers bertalign git+https://github.com/blagae/whitakers_words

Usage:
    python sentences/interlinear.py W05886 --out out/W05886                        # the whole book
    python sentences/interlinear.py W05886 --from 112.6 --to 114.3 --out out/W05886_locus5 --html
    python sentences/interlinear.py --src path/to/translation_en.json --out ...

Haiku runs through the Claude Code CLI on the subscription (common/via_claude.py), --jobs calls at a
time, a few sentence groups per call. Its answers are kept in <out>/haiku_cache.json, so a run that
stopped (a usage limit) goes on where it was when started again, and only asks for what changed.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from sentences.morph import Analyzer  # noqa: E402
from sentences.split import texts as sentences  # noqa: E402  (the one splitter, step 91)
from common.llm import MODEL_HAIKU  # noqa: E402
from common.works import id_of  # noqa: E402

MODEL = MODEL_HAIKU  # always Haiku (CLAUDE.md)
PROMPT_VERSION = "3"
ITALIC_RE = re.compile(r"<i>(.*?)</i>", re.S)
TAG_RE = re.compile(r"</?[a-z]+>")
GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
# A word (letters, with the -q; abbreviation of -que), a number, or one sign.
TOKEN_RE = re.compile(r"[^\W\d_]+q;|[^\W\d_]+(?:\[\?\][^\W\d_]*)*|\d+|[^\s\w]", re.U)
SUPERSCRIPT_SIGLA = "¹²³⁴⁵⁶⁷⁸⁹⁰ᵃᵇᶜᵈᵉᶠ"


# ---------------------------------------------------------------- the text

def split_unit(text: str) -> tuple[str, str, list[str]]:
    """(thesis number, running text, notes) of a unit's text.

    An italic span with a digit in it is a margin note of Scripture references; a single italic
    letter is a note sign in the text; any other italic span (a Greek word) stays in the text."""
    notes = []

    def italic(m):
        inner = m.group(1).strip()
        if re.fullmatch(r"[a-z]", inner):
            return " "
        if re.search(r"\d", inner) and len(inner) > 3:
            notes.append(TAG_RE.sub("", inner))
            return " "
        return inner

    body = ITALIC_RE.sub(italic, text)
    body = TAG_RE.sub("", body)
    body = re.sub(f"[{SUPERSCRIPT_SIGLA}]", " ", body)
    body = re.sub(r"\s+([,.;:])", r"\1", re.sub(r"\s+", " ", body)).strip()
    num = ""
    m = re.match(r"(\d+)\s+(?=\S)", body)
    if m:
        num, body = m.group(1), body[m.end():]
    return num, body, notes


def normalize(word: str) -> str:
    """The spelling Whitaker's Words looks up: no accents, ae for æ, -que for -q;, u for v."""
    w = word.replace("æ", "ae").replace("Æ", "Ae").replace("œ", "oe").replace("Œ", "Oe").replace("ſ", "s")
    w = re.sub(r"q;$", "que", w)
    w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
    return w.lower()


def tokens(sentence: str) -> list[dict]:
    out = []
    for m in TOKEN_RE.finditer(sentence):
        t = m.group(0)
        if t == "&":
            out.append({"form": t, "kind": "word", "norm": "et"})
        elif GREEK_RE.search(t):
            out.append({"form": t, "kind": "greek"})
        elif t[0].isalpha():
            out.append({"form": t, "kind": "word", "norm": normalize(t)})
        elif t.isdigit():
            out.append({"form": t, "kind": "number"})
        else:
            out.append({"form": t, "kind": "punct"})
    return out


# ---------------------------------------------------------------- sentences

class Aligner:
    """Bertalign with one LaBSE loaded once (on the GPU when torch has CUDA), and a cache of its
    pairings: <out>/align_cache.json, so a run started again does not pair the same sentences twice."""

    def __init__(self, model: str = "LaBSE", cache_file: Path | None = None):
        from bertalign import Bertalign  # noqa: F401  (import once, early, to fail fast)
        from bertalign.encoder import get_encoder
        self.encoder = get_encoder(model)
        self.cache_file = cache_file
        self.cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file and cache_file.exists() else {}
        self.new = 0

    def align(self, la: list[str], en: list[str]) -> list[tuple[list[int], list[int]]]:
        """Beads as (Latin sentence indices, English sentence indices); none of either side lost."""
        if len(la) == 1 or len(en) == 1:
            return [(list(range(len(la))), list(range(len(en))))]
        key = hashlib.sha1(json.dumps([la, en], ensure_ascii=False).encode()).hexdigest()
        if key not in self.cache:
            from bertalign import Bertalign
            a = Bertalign("\n".join(la), "\n".join(en), is_split=True, model=self.encoder,
                          top_k=min(3, len(en)), max_align=5)
            self.cache[key] = merge_empty([([int(i) for i in s], [int(i) for i in t]) for s, t in a.align_sents()])
            self.new += 1
            if self.new % 50 == 0:
                self.save()
        return [tuple(b) for b in self.cache[key]]

    def save(self):
        if self.cache_file:
            self.cache_file.write_text(json.dumps(self.cache), encoding="utf-8")


def merge_empty(beads):
    """A sentence Bertalign left without a partner joins the bead before it (or after it)."""
    out = []
    for s, t in beads:
        if out and (not s or not t):
            out[-1] = (out[-1][0] + s, out[-1][1] + t)
        else:
            out.append((s, t))
    if len(out) > 1 and (not out[0][0] or not out[0][1]):
        first = out.pop(0)
        out[0] = (first[0] + out[0][0], first[1] + out[0][1])
    return out




# ---------------------------------------------------------------- Haiku

SYSTEM = """You make an interlinear gloss of Latin for students. The text is from a Reformed
theological work of the 16th or 17th century; the English is a modern translation of it.

You get sentence groups, each with the Latin, the English, the Latin words numbered, each with its
candidate analyses from Whitaker's Words as "k lemma CODE" separated by |, and a lexicon with the
dictionary senses of each lemma.

Codes: V verb = V tense voice mood person number (tense P present, I imperfect, F future,
R perfect, L pluperfect, T future perfect; voice A active, P passive, D deponent; mood I indicative,
S subjunctive, M imperative, N infinitive (no person/number), P participle followed by -case number
gender). N noun = N C(common)|P(proper) case number gender. J adjective = J case number gender
(+C comparative, S superlative). R pronoun = R type(P personal, D demonstrative, R relative,
I interrogative/indefinite, X reflexive, A adjectival) - case number gender. Cases N G D A
B(ablative) V L; numbers S P; genders M F N C. B adverb, C conjunction, P preposition,
I interjection, M numeral, X other. A dash is an unknown slot.

For every numbered word give:

- pick: the number k of the candidate that is right in this context, or null if none is right
  (a printing error, a word Whitaker misses, a Greek word, a name).
- root: the sense of the lemma used here, as a short dictionary gloss of 1-3 words in its base
  form; a verb always with "to" ("to see", "to choose"), a noun in the singular ("will"), an
  adjective plain ("free"). Use the candidate's senses where they fit.
- gloss: a literal English rendering of this form, showing its grammar, 1-4 words, words joined
  with hyphens where they render one Latin word ("they-saw", "of-the-will", "to-be-chosen",
  "and", "freely", "souls"). For a verb include the person; for a noun or adjective show the
  case only where English needs a preposition (genitive "of-", dative "to-/for-", ablative
  "by-/with-/in-/from-" as the context needs); a nominative or accusative gets no mark at all.
  Never write case labels or brackets such as "(acc.)".
- en: the word or words of the given English sentence that render this Latin word, copied
  exactly; "" if the translation leaves it out.
- lemma and code: only when pick is null: your own dictionary form and code in the same system.
  For a printing error give the lemma of the intended word (igfallibiliter: infallibiliter).

A gerund (V-AG-case: "of willing") and a gerundive (V-PD-case number gender: "to be chosen") are
separate candidates: pick the one the sentence has.

Answer with JSON only: one list per group, in order, holding one list per numbered word, in order:
[i, pick, root, gloss, en] or, when pick is null, [i, null, root, gloss, en, lemma, code].
{"groups": [[[0, 1, "to see", "they-saw", "saw"], [1, null, "autexousion", "free-will", "free choice", "αὐτεξούσιον", "X"]], ...]}"""

WORDS_PER_CALL = 300   # Latin words in one Haiku call: a few sentence groups, an answer well under the output limit


def ask_haiku(groups: list[dict]) -> list[list]:
    from common import via_claude
    msg = json.dumps({"groups": groups}, ensure_ascii=False, indent=1)
    for attempt in range(3):
        if attempt and len(groups) > 4:
            # many short groups: Haiku loses count of them; two halves are asked apart and put together again
            half = len(groups) // 2
            print(f"   {len(groups)} groups asked in two halves", file=sys.stderr)
            return ask_haiku(groups[:half]) + ask_haiku(groups[half:])
        try:
            # $CLAUDE_CLI, else the native installer's place, else PATH (step 60); looked up here, so that a missing
            # one is a failed call, as it was
            cli = via_claude.find_cli(local_first=True)
            reply = via_claude.ask(SYSTEM, msg, MODEL, effort="low", cli=cli)   # glossing, not reasoning: low effort is faster
        except via_claude.UsageLimit:
            raise
        except RuntimeError as e:   # a failed CLI call: once more
            print(f"   Haiku call failed ({e}), again", file=sys.stderr)
            continue
        text = reply.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            data = json.loads(text[text.index("{"):text.rindex("}") + 1])
            answer = data["groups"]
            if len(answer) != len(groups):
                raise ValueError(f"{len(answer)} groups for {len(groups)}")
            for a, g in zip(answer, groups):
                if not isinstance(a, list) or len(a) < len(g["words"]):
                    raise ValueError(f"{len(a) if isinstance(a, list) else '?'} words for {len(g['words'])}")
            return answer
        except (ValueError, KeyError, TypeError) as e:
            print(f"   Haiku answer not readable ({e}), again", file=sys.stderr)
    raise RuntimeError("Haiku gave no readable answer three times")


# ---------------------------------------------------------------- the build

def prepare_unit(u, aligner, analyzer) -> dict:
    """The unit's beads (paired sentences), words with their candidates, and per bead what Haiku gets."""
    num, la_text, la_notes = split_unit(u["la"])
    _, en_text, en_notes = split_unit(u.get("tr") or "")
    la_s, en_s = sentences(la_text), sentences(en_text)
    beads = []
    for li, ei in (aligner.align(la_s, en_s) if la_s and en_s else [(list(range(len(la_s))), [])]):
        la = " ".join(la_s[i] for i in li)
        en = " ".join(en_s[i] for i in ei)
        toks = tokens(la)
        words, lexicon, n = [], {}, 0
        for t in toks:
            if t["kind"] not in ("word", "greek"):
                continue
            t["cands"] = analyzer.analyse(t["norm"]) if t["kind"] == "word" else []
            t["i"] = n
            options = []
            for k, c in enumerate(t["cands"]):
                options.append(f"{k} {c['lemma']} {c['code']}")
                senses = lexicon.setdefault(c["lemma"].split(" + ")[0], [])
                if c["senses"] and c["senses"][:150] not in senses and len(senses) < 4:
                    senses.append(c["senses"][:150])
            words.append(f"{n} {t['form']}: " + (" | ".join(options) if options else "(no candidates)"))
            n += 1
        beads.append({"la": la, "en": en, "tokens": toks,
                      "ask": {"latin": la, "english": en, "words": words, "lexicon": lexicon} if words else None})
    num = num or u.get("heading_num", "")
    return {"id": u["id"], "start": u["start"], "num": num, "beads": beads, "notes_la": la_notes, "notes_en": en_notes}


def calls_for(units) -> list[list[dict]]:
    """The beads that need Haiku, packed in book order into calls of about WORDS_PER_CALL words."""
    calls, cur, size = [], [], 0
    for u in units:
        for b in u["beads"]:
            if not b["ask"]:
                continue
            n = len(b["ask"]["words"])
            if cur and size + n > WORDS_PER_CALL:
                calls.append(cur)
                cur, size = [], 0
            cur.append(b)
            size += n
    if cur:
        calls.append(cur)
    return calls


def call_key(call) -> str:
    return hashlib.sha1((PROMPT_VERSION + MODEL + SYSTEM +
                         json.dumps([b["ask"] for b in call], ensure_ascii=False)).encode()).hexdigest()


def run_calls(calls, cache, cache_file, jobs) -> int:
    """Ask Haiku for every call not in the cache, jobs at a time; the cache is saved as answers come.
    Returns how many calls are still missing (after a usage limit or failures)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from common import via_claude
    todo = [c for c in calls if call_key(c) not in cache]
    print(f"{len(calls)} Haiku calls, {len(calls) - len(todo)} in the cache, {len(todo)} to ask ({jobs} at a time)", flush=True)
    lock, done, stop = threading.Lock(), [0], threading.Event()

    def one(call):
        if stop.is_set():
            return
        try:
            answer = ask_haiku([b["ask"] for b in call])
        except via_claude.UsageLimit as e:
            stop.set()
            print(f"   usage limit: {e}", file=sys.stderr, flush=True)
            return
        except RuntimeError as e:
            print(f"   {e}", file=sys.stderr, flush=True)
            return
        with lock:
            cache[call_key(call)] = answer
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == len(todo):
                cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                print(f"   {done[0]}/{len(todo)} calls answered", flush=True)

    with ThreadPoolExecutor(jobs) as pool:
        for f in as_completed([pool.submit(one, c) for c in todo]):
            f.result()
    cache_file.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return sum(1 for c in calls if call_key(c) not in cache)


def apply_answers(calls, cache):
    for call in calls:
        answer = cache.get(call_key(call))
        for b, words in zip(call, answer or [[] for _ in call]):
            words = [w for w in words if isinstance(w, list) and w]
            # one answer per word, in order: by place (Haiku sometimes slips in the number it writes first);
            # by that number only when the count is off
            n_words = sum(1 for t in b["tokens"] if "i" in t)
            by_i = dict(enumerate(words)) if len(words) == n_words else {w[0]: w for w in words}
            for t in b["tokens"]:
                if "i" not in t:
                    continue
                a = by_i.get(t["i"], [])
                a = list(a) + [None] * (7 - len(a))
                _, pick, root, gloss, en, lemma, code = a[:7]
                cands = t.pop("cands", [])
                if isinstance(pick, int) and 0 <= pick < len(cands):
                    c = cands[pick]
                    t.update(lemma=c["lemma"], code=c["code"], senses=c["senses"], source="whitaker")
                elif answer is not None:
                    t.update(lemma=lemma or "", code=code or "X", senses="", source="haiku")
                t.update(n_cands=len(cands), root=root or "", gloss=gloss or "", en=en or "")
                t.pop("i")
            b.pop("ask", None)


def relabel(doc, analyzer) -> tuple[int, int]:
    """An earlier result put through the dictionary as it is now (morph.py improved), without asking
    Haiku again: Haiku's own analysis that the dictionary now has counts as the dictionary's (esse, sum,
    VPAN), and a dictionary analysis gets the root it now has (fuerit: sum, no longer fuerit).
    Returns (Haiku's own now in the dictionary, roots changed)."""
    def same(a, b):
        f = lambda s: (s or "").lower().replace("j", "i").replace("v", "u").split(" + ")[0]
        return f(a) == f(b)
    found = fixed = 0
    for u in doc["units"]:
        for b in u["beads"]:
            for t in b["tokens"]:
                if t["kind"] != "word" or "norm" not in t:
                    continue
                cands = analyzer.analyse(t["norm"])
                if t.get("source") == "haiku":
                    c = next((c for c in cands if c["code"] == t.get("code") and same(c["lemma"], t.get("lemma"))), None)
                    if c:
                        t.update(lemma=c["lemma"], senses=c["senses"], source="whitaker")
                        found += 1
                elif t.get("source") == "whitaker":
                    c = next((c for c in cands if c["code"] == t.get("code") and c["senses"] == t.get("senses")), None)
                    if c and c["lemma"] != t["lemma"]:
                        t["lemma"] = c["lemma"]
                        fixed += 1
                t["n_cands"] = len(cands)
    return found, fixed


def load_units(data, first: str | None, last: str | None):
    units, on, heading_num = [], first is None, ""
    for u in data["units"]:
        m = re.match(r"\s*Thesis\s+(\d+)", u["la"]) if u.get("heading", 0) else None
        if m:  # "Thesis 1." stands as a heading of its own above the first thesis
            heading_num = m.group(1)
        on = on or u["id"] == first
        if on and u.get("heading", 0) == 0 and (u.get("la") or "").strip():
            units.append(dict(u, heading_num=heading_num))
            heading_num = ""
        if u["id"] == last:
            break
    return units


def find_src(work: str, lang: str) -> Path:
    hits = sorted((BACKEND / "data").glob(f"{work}_*/translation_{lang}.json"))
    if not hits:
        sys.exit(f"no translation_{lang}.json for {work} under {BACKEND / 'data'}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work", nargs="?", help="work number, such as W05886")
    ap.add_argument("--src", type=Path, help="translation_<lang>.json (instead of the work number)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--from", dest="first", help="first unit id, such as 112.6 (default: the first)")
    ap.add_argument("--to", dest="last", help="last unit id (default: the last)")
    ap.add_argument("--title", default="")
    ap.add_argument("--jobs", type=int, default=3, help="Haiku calls at a time (default 3)")
    ap.add_argument("--html", action="store_true", help="also the page of its own (interlinear.html)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-haiku", action="store_true",
                    help="steps 1 and 2 only: pair the sentences (align_cache.json) and look the words up, and write in "
                         "prep.json what Haiku would get (units, beads, words, calls), without asking it")
    ap.add_argument("--relabel", action="store_true",
                    help="only put <out>/interlinear.json through the dictionary as it is now, and write the reader's copy again")
    args = ap.parse_args()
    if args.relabel:
        doc = json.loads((args.out / "interlinear.json").read_text(encoding="utf-8"))
        found, fixed = relabel(doc, Analyzer())
        (args.out / "interlinear.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        site = write_site(doc, args.out / "site")
        own = sum(1 for u in doc["units"] for b in u["beads"] for t in b["tokens"] if t.get("source") == "haiku")
        print(f"{found} of Haiku's own analyses now from the dictionary, {fixed} roots corrected; "
              f"{own} still Haiku's own; written {site}")
        return
    args.work = args.work and id_of(args.work)
    src = args.src or find_src(args.work, args.lang)
    data = json.loads(src.read_text(encoding="utf-8"))
    units = load_units(data, args.first, args.last)
    if not units:
        sys.exit(f"no units from {args.first} to {args.last} in {src}")
    work = args.work or (re.match(r"W\d+", src.parent.name) or re.match(r"W\d+", str(data.get("book", ""))) or [""])[0]
    args.out.mkdir(parents=True, exist_ok=True)
    cache_file = args.out / "haiku_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    print(f"{work}: {len(units)} units; loading LaBSE and Whitaker's Words", flush=True)
    aligner, analyzer = Aligner(cache_file=args.out / "align_cache.json"), Analyzer()
    try:
        import torch
        where = "the GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "the CPU"
    except ImportError:          # a stand-in for Bertalign (the tests)
        where = "no torch"
    print(f"   LaBSE on {where}; {len(aligner.cache)} pairings in the cache", flush=True)
    built = []
    for n, u in enumerate(units, 1):
        built.append(prepare_unit(u, aligner, analyzer))
        if n % 100 == 0 or n == len(units):
            print(f"   {n}/{len(units)} units aligned and analysed", flush=True)
    aligner.save()
    calls = calls_for(built)
    if args.no_haiku:
        prep = prep_summary(work or data.get("book"), args.lang, built, calls)
        (args.out / "prep.json").write_text(json.dumps(prep, ensure_ascii=False), encoding="utf-8")
        print(f"steps 1 and 2 done: {prep['beads']} beads, {prep['words']} words "
              f"({prep['no_candidates']} without candidates); Haiku would get {prep['calls']} calls")
        return
    missing = run_calls(calls, cache, cache_file, args.jobs)
    apply_answers(calls, cache)

    doc = {"work": work or data.get("book"), "lang": args.lang, "from": units[0]["id"], "to": units[-1]["id"],
           "title": args.title, "model": MODEL, "aligner": "bertalign/LaBSE", "morphology": "whitakers_words",
           "complete": missing == 0, "units": built}
    (args.out / "interlinear.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.html:
        from sentences.render import render
        (args.out / "interlinear.html").write_text(render(doc), encoding="utf-8")
    site = write_site(doc, args.out / "site")
    words = sum(1 for u in built for b in u["beads"] for t in b["tokens"] if t["kind"] in ("word", "greek"))
    own = sum(1 for u in built for b in u["beads"] for t in b["tokens"] if t.get("source") == "haiku")
    print(f"written: {args.out / 'interlinear.json'} and {site} ({words} words, {own} analysed by Haiku itself)")
    if missing:
        print(f"{missing} of {len(calls)} Haiku calls still missing: run the same command again later")
    print(f"for the reader: copy {site} to frontend/data/interlinear/{doc['work']}/{doc['lang']} "
          f"and add the work to frontend/data/interlinear/index.json")


def prep_summary(work, lang, built, calls) -> dict:
    """What steps 1 and 2 gave, and what step 3 would ask of Haiku (prep.json of --no-haiku)."""
    toks = [t for u in built for b in u["beads"] for t in b["tokens"] if t["kind"] in ("word", "greek")]
    return {"work": work, "lang": lang, "units": len(built), "beads": sum(len(u["beads"]) for u in built),
            "words": len(toks), "no_candidates": sum(1 for t in toks if t["kind"] == "word" and not t.get("cands")),
            "calls": len(calls), "words_per_call": WORDS_PER_CALL, "aligner": "bertalign/LaBSE",
            "morphology": "whitakers_words"}


SCANS_PER_FILE = 10   # the reader fetches a file for each ten scans that come into view


def write_site(doc, root: Path) -> Path:
    """The reader's copy: frontend/data/interlinear/<work>/<lang>/index.json and <n>.json for scans
    10n to 10n+9. Per unit its first line [scan, line] (the reader finds the translated paragraph by
    it), per word [form, lemma, code, root, gloss, English word, 1 if Haiku's own analysis], a sign of
    punctuation as [sign]; the dictionary senses once per lemma in the file's lexicon."""
    import shutil
    out = root / doc["work"] / doc["lang"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {}
    for u in doc["units"]:
        f = files.setdefault(u["start"][0] // SCANS_PER_FILE, {"lex": {}, "units": []})
        beads = []
        for b in u["beads"]:
            ws = []
            for t in b["tokens"]:
                if t["kind"] in ("punct", "number"):
                    ws.append([t["form"]])
                    continue
                if t.get("senses"):
                    f["lex"].setdefault(t["lemma"], t["senses"])
                ws.append([t["form"], t.get("lemma", ""), t.get("code", ""), t.get("root", ""), t.get("gloss", ""),
                           t.get("en", ""), 1 if t.get("source") == "haiku" else 0])
            beads.append({"la": b["la"], "en": b["en"], "w": ws})
        f["units"].append({"id": u["id"], "start": u["start"], "num": u["num"], "beads": beads, "notes": u["notes_la"]})
    for n, f in files.items():
        (out / f"{n}.json").write_text(json.dumps(f, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    meta = {k: doc[k] for k in ("work", "lang", "model", "aligner", "morphology", "complete")}
    meta.update(scans_per_file=SCANS_PER_FILE, files=sorted(files), units=len(doc["units"]))
    (out / "index.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
