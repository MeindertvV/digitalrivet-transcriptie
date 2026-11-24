"""The gate against duplicates (backend/docs/DUBBELEN.md, step 0 and option C).

Every expensive step asks here first whether a work has to wait because it is (probably) the same book as another
work in the library. Works stay in the library, also twice: the gate only keeps the same book from being
transcribed or translated twice.

Transcription (held):
  With the candidates of duplicates/scan.py (enrich/dubbelen_kandidaten.csv), the rule per group: works joined by
  open pairs (a candidate without a closing decision, in any band but "waarschijnlijk verschillend" or "zwak") form a group;
  a work without book.json waits if a work in its group is already transcribed, or else if another work of the group
  goes first. So at most one work per group starts. Goes first: the reading edition a decision names, else a work not
  in special_layout.tsv, then the most pages, then the lowest number (W03293 before W03293-02). In a group of more
  than CAP works (mostly a general title) the rule is per pair.
  Without that file: the fixed list of step 0, transcription/vasthouden_dubbel.tsv.
Translation (not_to_translate):
  translation/niet_vertalen.tsv; a work a decision calls the same scan or the same druk as another work, if it is
  not the reading edition; and a work of an open pair in band "zelfde scan" or "waarschijnlijk zelfde werk" whose
  twin is already translated.

Decisions (PRDL/dubbelen_besluiten.csv, written by duplicates/screen.py): verschillend, deel_van and bundel close a
pair; zelfde_scan, zelfde_druk and andere_druk keep it (one of the two is enough); verkeerde_scan holds that work until
there is a right scan. --ook-dubbel on transcribe.py or translate.py overrides all this. A work that already has a
book.json is never held for transcription: its repairs go on.
"""

import csv
import functools
import io
import os
import time
from collections import defaultdict
from pathlib import Path

from common import paths  # noqa: E402
from common.languages import TRANSLATED  # noqa: E402
from common.paths import DATA, PDF_DIR  # noqa: E402  (PRDL_DATA for a test copy)
from common.works import WORK_ID  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
# The checkout that owns the books (the dev copy links backend/data to the live one): its candidates, the list of
# what the scan saw and the owner's decisions hold for every checkout, so a run from dev follows the same gate
HOME = paths.owner()
HOLD = paths.held_duplicates(HOME)
NO_TRANSLATE = paths.no_translate(HOME)
CANDIDATES = paths.duplicate_candidates(HOME)
SEEN = paths.duplicates_seen(HOME)   # the works the last full scan saw, with their PDF
DECISIONS = paths.duplicate_decisions(HOME)
SPECIAL = paths.special_layout(HOME)
CAP = 10
LANGS = set(TRANSLATED)   # the languages of translate.py (translation_oc.json is the Opera Calvini, not a translation)
CLOSING = {"verschillend", "deel_van", "bundel"}
SAME = {"zelfde_scan", "zelfde_druk", "andere_druk"}
KINDS = CLOSING | SAME | {"verkeerde_scan"}



def work_id(name: str | Path) -> str | None:
    """'W08036_Author_Title_1626' (a PDF, a book folder or a bare id) -> 'W08036'; a volume keeps its part: 'W07610-02'."""
    m = WORK_ID.match(Path(str(name)).name)
    return m.group() if m else None


def listed(path: Path) -> dict[str, tuple[str, str]]:
    """{work: (reason, twins)} from a tab-separated list; '#' lines are comments. A missing list holds nothing."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t") + ["", ""]
        out[f[0].strip()] = (f[1].strip(), f[2].strip())
    return out


def csv_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")), delimiter=";"))


def decision_rows(path: Path, kinds: set | None) -> list[dict]:
    """The rows of a file of decisions (dubbelen_besluiten.csv); none when it is not there. With kinds, a row of
    another kind stops everything; with None every row comes as it is. The one reader of the gate, the scan
    (duplicates/scan.py) and the catalogue (build_frontpage.py), each with its own file."""
    if not path.exists():
        return []
    rows = csv_rows(path)
    for r in rows if kinds is not None else ():
        if (r.get("soort") or "").strip() not in kinds:
            raise ValueError(f"{path.name}: unknown kind {r.get('soort')!r} in {r}")
    return rows


def decisions() -> list[dict]:
    """The rows of dubbelen_besluiten.csv; a row with an unknown kind stops everything (a typo must not open a gate)."""
    return decision_rows(DECISIONS, KINDS)


def fresh_state() -> dict:
    """What _state() gives, read again (the screen of duplicates/screen.py, after a decision)."""
    _state.cache_clear()
    return _state()


@functools.lru_cache(maxsize=1)
def _state() -> dict:
    """Everything the gate needs, read once per process."""
    folders = {}
    if DATA.exists():
        for n in os.listdir(DATA):
            if (w := work_id(n)):
                folders[w] = DATA / n
    done = {w for w, f in folders.items() if (f / "book.json").exists()}
    # started: paid pages (raw/) or a partial book, no book.json yet. It is never held (its batches are paid for), and
    # for its twins it counts as done
    started = {w for w, f in folders.items() if w not in done and (
        (f / "partial").exists() or (f / "batches.json").exists() or next(f.glob("raw/p*.json"), None)
        or next(f.glob("pages/*.xml"), None))}   # submitted batches count: paid for, results still to come
    pdfs = {}
    for n in (os.listdir(PDF_DIR) if PDF_DIR.exists() else []):
        if n.endswith(".pdf") and (w := work_id(n)):
            pdfs[w] = PDF_DIR / n
    # translated: a translation file. Begun: paid units in translation_*/raw/ (may be finished). An empty folder of a
    # failed run (no credit) is neither
    # per language: {lang: works}
    tr_done, tr_begun = defaultdict(set), defaultdict(set)
    for w, f in folders.items():
        for t in f.glob("translation_*.json"):
            lang = t.stem.split("_", 1)[1]
            if lang in LANGS:
                tr_done[lang].add(w)
                tr_begun[lang].add(w)
        for t in f.glob("translation_*"):
            lang = t.name.split("_", 1)[1]
            # begun: units fetched (raw/) or a batch submitted and paid for (batches.json), results still to come
            if t.is_dir() and lang in LANGS and (next(t.glob("raw/*"), None) or (t / "batches.json").exists()):
                tr_begun[lang].add(w)
    translated = set().union(*tr_done.values()) if tr_done else set()
    special = set(listed(SPECIAL)) if SPECIAL.exists() else set()
    st = dict(folders=folders, done=done | started, started=started, open=set(pdfs) - done - started, pdfs=pdfs,
              translated=translated, tr_done=tr_done, tr_begun=tr_begun, special=special, source="", edges=defaultdict(set), pages={},
              strong=defaultdict(set), wrong=set(), reading_named=set(), same=defaultdict(set), le={}, closed=set(),
              unscanned=set())
    last = {}   # the last decision for a pair replaces the earlier ones
    for r in decisions():
        a, b = (r.get("werk") or "").strip(), (r.get("ander") or "").strip()
        key = tuple(sorted((a, b))) if b else (a, "")
        last.pop(key, None)   # re-inserted: the order is that of the latest decisions
        last[key] = r
    st["decided"] = {k for k in last if k[1]}
    for r in last.values():
        a, b, kind = (r.get("werk") or "").strip(), (r.get("ander") or "").strip(), r["soort"].strip()
        if kind == "verkeerde_scan":
            st["wrong"].add(a)
            if b:
                st["closed"].add(tuple(sorted((a, b))))   # the other work is not held by a wrong scan
        if not b:
            continue
        pair = tuple(sorted((a, b)))
        if kind in CLOSING:
            st["closed"].add(pair)
        elif kind in SAME:
            st["edges"][a].add(b)
            st["edges"][b].add(a)
            st["same"][a].add(b)
            st["same"][b].add(a)
            if (le := (r.get("leeseditie") or "").strip()):
                st["le"][pair] = le          # in the order of the latest decisions
    st["reading_named"] = set(st["le"].values())
    if CANDIDATES.exists():
        try:
            rows = csv_rows(CANDIDATES)
            for r in rows:
                st["pages"][r["a"]] = int(r.get("pagina_a") or 0)
                st["pages"][r["b"]] = int(r.get("pagina_b") or 0)
            st["source"] = "kandidaten"
            age = (time.time() - CANDIDATES.stat().st_mtime) / 3600
            if age > 48:   # the nightly scan did not run or failed: new PDFs wait, and nobody would notice
                print(f"common/dubbel.py: let op, {CANDIDATES.name} is {age:.0f} uur oud (./rivet dubbelen_nacht, "
                      f"backend/logs/DUBBELEN_SCAN_MISLUKT.txt)", flush=True)
        except Exception as e:   # a broken list must not open the gate: stop, as for a broken list of decisions
            raise ValueError(f"common/dubbel.py: {CANDIDATES} not readable ({e}); run enrich/dubbelen.py again") from e
        for r in rows:
            a, b = r["a"], r["b"]
            if tuple(sorted((a, b))) in st["closed"] or r["band"] in ("waarschijnlijk verschillend", "zwak"):
                continue
            st["edges"][a].add(b)
            st["edges"][b].add(a)
            if r["band"] in ("zelfde scan", "waarschijnlijk zelfde werk"):
                st["strong"][a].add(b)
                st["strong"][b].add(a)
        if st["source"]:
            # an open work the last full scan did not see (SEEN: "W00001 t|o|-", the text it had), or that has text
            # now while the scan saw none. Without SEEN: its PDF newer than the candidates
            seen = None
            if SEEN.exists():   # "W00001 t|o|- size mtime" (older lists: without the PDF)
                seen = {f[0]: f[1:] for ln in SEEN.read_text(encoding="utf-8").splitlines() if (f := ln.split())}
            since = CANDIDATES.stat().st_mtime
            for w in st["open"]:
                f = folders.get(w)
                try:
                    st_ = pdfs[w].stat()
                except OSError:   # moved away (to _dubbel/) since the folder was read: not seen
                    st["unscanned"].add(w)
                    continue
                if seen is None:
                    new = st_.st_mtime > since
                else:
                    k = (seen.get(w) or ["-"]) + ["", ""]
                    new = (w not in seen or (k[0] == "-" and bool(f) and (f / "ocr_layer" / "book.json").exists())
                           or (k[1] and (k[1], k[2]) != (str(st_.st_size), str(int(st_.st_mtime)))))   # PDF replaced
                if new:
                    st["unscanned"].add(w)
    # a work that can start: done, or a PDF that is not a wrong scan and not waiting on the list of step 0
    st["can_start"] = st["done"] | {w for w in st["open"] - st["wrong"] if not _step0(w, st)}
    st["hold"] = _groups(st)   # without candidates: the decisions alone (and the list of step 0, in held)
    return st


def _num(w: str) -> tuple[int, int]:
    return int(w[1:6]), int(w[7:]) if len(w) > 6 else 0


def _components(nodes: set, edges: dict) -> list[set]:
    seen, out = set(), []
    for start in sorted(nodes):
        if start in seen:
            continue
        group, todo = set(), [start]
        while todo:
            w = todo.pop()
            if w in group:
                continue
            group.add(w)
            todo.extend((edges[w] & nodes) - group)
        seen |= group
        out.append(group)
    return out


def _groups(st: dict) -> dict[str, str]:
    """{work: reason} for every open work that waits.
    1. Works a decision calls the same book (1-3), as a chain: if one of them is done or started, the others wait;
       else only the reading edition (the named one, else the rule of reading_edition) may start.
    2. A twin (a direct candidate pair) is done or started: wait. Through a third work only when that one has no
       PDF (two scans of one title record).
    3. A wrong scan, and a work the last scan did not read, wait; they never go first for others.
    4. Else, of the works that are not done and hang together through open pairs, one goes first (also through a
       work without PDF: two scans of one title twin). In a group of more than CAP, per pair."""
    edges, done, opn = st["edges"], st["done"], st["open"]
    prio = lambda w: (w in st["reading_named"], w not in st["special"] and w.split("-")[0] not in st["special"],
                      st["pages"].get(w, 0), tuple(-n for n in _num(w)))
    hold = {}
    for group in _components(set(st["same"]), st["same"]):
        d = sorted(group & done)
        le = _reading(next(iter(group)), st)
        for w in group & opn:
            if d:
                hold[w] = f"zelfde boek als {', '.join(d)} (besluit), dat al {'klaar' if set(d) - st['started'] else 'begonnen'} is"
            elif w != le:
                hold[w] = f"zelfde boek als {', '.join(sorted(group - {w}))} (besluit); de leeseditie {le} gaat voor"
    for w in opn:
        if w not in hold and (d := sorted(edges[w] & done)):
            hold[w] = f"tweeling {'klaar' if set(d) - st['started'] else 'bezig'}: {', '.join(d)}"
    # through a record without PDF (two scans of one title twin, as Antesignan W00293/W00295 via W00296): the same
    for w in opn:
        if w in hold:
            continue
        for n in sorted(edges[w]):
            if n not in st["pdfs"] and n not in done and (d := sorted((edges[n] & done) - {w})):
                hold[w] = f"tweeling {'klaar' if set(d) - st['started'] else 'bezig'}: {', '.join(d)} (via {n}, zonder PDF)"
                break
    for w in st["wrong"] & opn:
        hold[w] = "verkeerde scan (besluit)"
    for w in st["unscanned"]:
        hold.setdefault(w, "nog niet bekeken door enrich/dubbelen.py (nieuwe PDF of nog geen tekstlaag)")
    rest = {w for w in edges if w not in done}
    for group in _components(rest, edges):
        free = [w for w in group if w in opn and w not in hold and w in st["can_start"]]
        if len(free) < 2:
            continue
        if len(group) > CAP:   # per pair, from the first down: a work goes unless a direct twin above it goes
            going = set()
            # twins: direct pairs, and two scans of one title record without PDF (they share that neighbour)
            near = lambda w: edges[w] | {x for n in edges[w] if n not in st["pdfs"] and n not in done for x in edges[n]}
            for w in sorted(free, key=prio, reverse=True):
                if (o := sorted((n for n in near(w) if n in going and n != w), key=prio)):
                    hold[w] = f"open tweeling gaat voor: {o[-1]}"
                else:
                    going.add(w)
            continue
        first = max(free, key=prio)
        for w in free:
            if w != first:
                hold[w] = f"in de groep gaat voor: {first}"
    return hold


def _step0(w: str, st: dict) -> str | None:
    """The line of step 0 for w, unless every pair it names has a decision (then the rules of the decision hold)."""
    hit = listed(HOLD).get(w)
    if not hit:
        return None
    reason, twins = hit
    tw = [t.strip("*") for t in twins.split(",") if t.strip()]
    chain = next((g for g in _components(set(st["same"]), st["same"]) if w in g), {w})   # decided as one book with w
    tw = [t for t in tw if tuple(sorted((w, t))) not in st["decided"] and t not in chain]
    if reason.startswith("open tweeling") and st["source"]:
        tw = [t for t in tw if t not in st["edges"][w]]   # a pair the scan also proposes: the group rule decides who goes
    if not tw:
        return None
    if reason.startswith("open tweeling") and not any(t in st["done"] for t in tw):
        # "the open twin goes first": only while one of them can go first. Of twins that name each other, the one
        # with the most pages (then the lowest number) goes; a twin without PDF goes nowhere
        rank = lambda x: (st["pages"].get(x, 0), tuple(-n for n in _num(x)))
        if not any(t in st["open"] and t not in st["wrong"] and rank(t) > rank(w) for t in tw):
            return None
    return f"{w}: lijst van stap 0, {reason}" + (f" (tweelingen {twins.replace('*', '')})" if twins else "") + \
        f"; zie transcription/{HOLD.name}"


def held(name: str | Path) -> str | None:
    """Why this work may not be transcribed now, or None. Never for a work with a transcription (its repairs go on) or
    one already started (its batches are paid for). The list of step 0 holds as well, until a decision."""
    w = work_id(name)
    if not w:
        return None
    st = _state()
    if w in st["done"]:
        return None
    if (r := st["hold"].get(w)):
        return f"{w}: {r}; zie enrich/{CANDIDATES.name} en een besluit in tools/dubbelen_server.py"
    return _step0(w, st)


def reading_edition(w: str) -> str | None:
    """The reading edition of w's book (see _reading)."""
    return _reading(w, _state())


def _reading(w: str, st: dict) -> str | None:
    """The reading edition of the works decisions call the same book as w (zelfde scan, zelfde druk, andere druk; as a
    chain): the one a decision names (the last), as long as it is transcribed when another one is; else a translated
    one, then a transcribed one, then the most pages, then the lowest number. None if there is no such decision."""
    if not st["same"].get(w):
        return None
    group = next(g for g in _components(set(st["same"]), st["same"]) if w in g)
    done = group & st["done"] - st["started"]
    able = (group - st["wrong"]) & st.get("can_start", group) or (group - st["wrong"]) or group   # one that can start
    named = [le for pair, le in st["le"].items() if pair[0] in group and le in able and (le in done or not done)]
    if named:
        return named[-1]
    special = lambda x: x in st["special"] or x.split("-")[0] in st["special"]   # budget_run.py does not start those
    nv = listed(NO_TRANSLATE)
    return max(able, key=lambda x: (x not in nv, x in st["translated"], x in done, not special(x), st["pages"].get(x, 0),
                                    tuple(-n for n in _num(x))))


def not_to_translate(name: str | Path, lang: str | None = None) -> str | None:
    """Why this work may not be translated (into lang), or None. A work whose translation into that language has
    begun may finish it; a translation into another language does not count."""
    w = work_id(name)
    if not w:
        return None
    st = _state()
    begun = st["tr_begun"].get(lang, set()) if lang else set().union(*st["tr_begun"].values()) if st["tr_begun"] else set()
    if w in begun:   # paid units: may be finished
        return None
    hit = listed(NO_TRANSLATE).get(w)
    if hit:   # until a decision on the pair (then its reading edition holds, as for the list of step 0)
        reason, twins = hit
        tw = [t.strip() for t in twins.split(",") if t.strip()]
        if not (tw and all(tuple(sorted((w, t))) in st["decided"] or t in st["same"].get(w, set()) for t in tw)):
            return f"{w}: {reason}" + (f" (leeseditie {twins})" if twins else "") + f"; zie translation/{NO_TRANSLATE.name}"
    if w in st["wrong"]:
        return f"{w}: verkeerde scan (besluit)"
    if st["same"].get(w):   # decided as one book: per language only one copy is translated
        chain = next(g for g in _components(set(st["same"]), st["same"]) if w in g)
        if (b := sorted(chain & begun)):
            return f"{w}: zelfde boek als {', '.join(b)} (besluit), dat al in deze taal vertaald (of begonnen) is"
        le = reading_edition(w)
        if le and le != w:
            return f"{w}: zelfde boek als {', '.join(sorted(st['same'][w]))} (besluit); alleen de leeseditie {le} wordt vertaald"
        return None   # the decision settles it: the candidate rules below do not apply
    strong = st["strong"].get(w, set())
    if (t := sorted(strong & begun)):
        return f"{w}: waarschijnlijk zelfde boek als {', '.join(t)}, dat al vertaald is; eerst een besluit"
    # of two probable twins that are both untranslated, one goes first (transcribed, then most pages, lowest number)
    tp = lambda x: (x in st["done"] and x not in st["started"], st["pages"].get(x, 0), tuple(-n for n in _num(x)))
    # a rival only counts if it may be translated itself: else this one would wait for nothing
    free = lambda n: n not in st["wrong"] and n not in listed(NO_TRANSLATE) and not (st["strong"].get(n, set()) & begun) \
        and (not (le := _reading(n, st)) or le == n)
    rivals = [n for n in strong if tp(n) > tp(w) and free(n)]
    if (o := sorted(rivals, key=tp)):
        return f"{w}: waarschijnlijk zelfde boek als {o[-1]}, dat voorgaat bij het vertalen; eerst een besluit"
    return None
