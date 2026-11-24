"""The contents of a book, from one request to Claude Haiku on a sheet with context.

toc_haiku.py reads a sheet of only the lines that may be headings, in parts, and repairs the result
in many passes and rules. Here the model gets the whole sheet at once, with the first words of the
text after those lines and how much text that is (sheet()), says first what the book is and which
series it must have, and names what it cannot find on the sheet. So there are three steps only:
  1. one answer for the whole book;
  2. the checks that keep it honest: only lines of the sheet, in the book's order, a title that
     belongs to its lines (checked());
  3. a look at the scans for what the model missed or a title that did not hold, and nothing else
     (looked()).
tally() then counts the answer against what the model itself promised, so that an outline that fell
short shows without anyone reading the book (doubted()).

The result has the form of toc_haiku.json and stands in data/<work>/toc_context.json, the answers in
data/<work>/toc_context/raw/; --out writes them elsewhere, for a trial. contents/structure.py takes this
outline before that of toc_haiku.py, unless the file holds a doubt. The questions go through the
`claude` command line, on the subscription, not with the API key.

Examples
  python toc_context.py W05293
  python toc_context.py W05293 --out ~/.cache/toc_context/run --sheet
  python toc_context.py --list works.txt --parallel 3        # a book that has its outline is skipped (--redo)
Run it as env -u ANTHROPIC_API_KEY CLAUDE_CLI=~/.local/bin/claude ...; at the subscription's usage
limit the run stops asking, and run again it goes on where it stopped.
"""

import argparse
import base64
import bisect
import datetime as dt
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.via_claude import LIMIT_RE, UsageLimit, call  # noqa: E402
from common.paths import CLAUDE_LOCAL, PROMPTS  # noqa: E402
from common.llm import MODEL_HAIKU  # noqa: E402
from contents import headings as T  # noqa: E402  (was toc_haiku: REFACTOR_PLAN.md steps 36 and 37)

CLAUDE = Path(os.environ.get("CLAUDE_CLI") or CLAUDE_LOCAL)   # the native installer's place (common/paths.py)
MODEL = MODEL_HAIKU   # the full id: it goes into the key of every kept answer (step 58)
EFFORT = "medium"
PROMPT = PROMPTS / "toc_context.txt"
SCAN_PROMPT = PROMPTS / "toc_context_scan.txt"
OUT_NAME = "toc_context.json"
RAW = "toc_context/raw"
TASTES = (50, 25, 12, 6)    # words of the text shown after a line: the most that fits BUDGET
BUDGET = 95000              # tokens of a request; above 100,000 the model costs five times as much
CHARS = 1.8                 # characters of a sheet per token
WHOLE = BUDGET              # tokens up to which a book goes in one request; above that, in one request the model gave only the top of a thick volume (Lightfoot: works without their chapters)
CARRY = 12                  # entries of the outline so far given with the next part of a book that does not fit
TIMEOUT = 1800              # seconds an answer may take
LOOK_SCANS = 6              # scans at most shown to find one division
LOOK_MOST = 12              # divisions at most looked for per book
LOOK_PX = 1000              # long edge of a scan shown
NEARBY = 2                  # scans around a line where the line a title came from is looked for
SHORT = 0.6                 # share of a promised series below which the outline is doubted
SHORT_MIN = 5               # members of a series from which that says something
EMPTY = 0.5                 # share of a book's scans without an entry from which the outline is doubted
EMPTY_MIN = 100             # scans from which that says something
LIMITED = threading.Event()   # the subscription's limit was reached: every later question fails at once
LINE_NUMBER = re.compile(r"^(\d+) l\d+ ")
NUMBER = re.compile(r"\b(?:[IVXLCDMJ]{1,8}|\d{1,4})\b", re.I)
LABELLED = re.compile(r"(?:\b(?:liber|lib|caput|capvt|cap|capitulum|pars|sectio|sect|quaestio|quaest|disputatio|disp|"
                      r"dissertatio|exercitatio|articulus|art|locus|tractatus|homilia|sermo|concio|oratio|epistola|epist|"
                      r"psalmus|thesis|tomus|titulus|tit|regula|appendix)|§)\W{0,3}([IVXLCDMJ]{1,8}|\d{1,4})\b", re.I)


def text_of(root) -> tuple[list[tuple[int, str]], dict[int, int]]:
    """The book as [(scan, line)], and where each scan begins in that list."""
    seq, first = [], {}
    for pg in root.iter("page"):
        scan = int(pg.get("n"))
        first[scan] = len(seq)
        seq += [(scan, " ".join(T.R.plain(l).split())) for l in T.lines_of(pg)]
        seq += [(scan, " ".join(T.R.plain(h).split())) for h in pg.findall("head")]
    return seq, first


def places(cands: list[dict], seq, first) -> dict:
    """Where each candidate stands in the text. Lines can lack a number, so they are found by their
    words, going on from the last one; a running head stands above its page."""
    at, cursor = {}, 0
    for c in cands:
        cursor = max(cursor, first[c["scan"]])
        at[c["id"]] = first[c["scan"]] - 1 if c["level"] == -2 else cursor
        if c["level"] == -2:
            continue
        for i in range(cursor, len(seq)):
            if seq[i][0] != c["scan"]:
                break
            if seq[i][1][:T.CUT] == c["text"]:
                at[c["id"]] = cursor = i
                break
    return at


def sheet(root, cands: list[dict], taste: int) -> list[str]:
    """The sheet of toc_haiku, without line numbers, with after a line the words that follow it:
    only where text follows, so that the lines of one heading stay together. ^ marks the last line
    of a page, which is often a catchword."""
    seq, first = text_of(root)
    at = places(cands, seq, first)
    stops = sorted(set(at.values()))
    by_id = {c["id"]: c for c in cands}
    out = []
    for line in T.sheet(root, cands):
        if line.startswith("=="):
            out.append(line)
            continue
        c = by_id[int(line.split()[0])]
        out.append(LINE_NUMBER.sub(r"\1 ", line) + (" ^" if c.get("last") else ""))
        p = at[c["id"]]
        k = bisect.bisect_right(stops, p)
        end = stops[k] if k < len(stops) else len(seq)
        words = " ".join(t for _, t in seq[p + 1:end]).split()
        if words:
            out.append(f"> {' '.join(words[:taste])} [{len(words)} w]")
    return out


def request(root, folder: Path, cands: list[dict]) -> tuple[str, int]:
    """The text for the model, with as many words after a line as fit, and that number."""
    seq, _ = text_of(root)
    head = (f"Book: {root.get('source', folder.name).removesuffix('.pdf')}\n{sum(1 for _ in root.iter('page'))} scans, "
            f"{sum(len(t.split()) for _, t in seq)} words.\n\n")
    printed = T.contents_pages(root)
    if printed:
        head += f"Text of pages that may be the book's printed table of contents (may also be another kind of index, then ignore it):\n{printed}\n\n"
    for taste in TASTES:
        text = head + "Lines:\n" + "\n".join(sheet(root, cands, taste))
        if len(text) / CHARS <= BUDGET:
            break
    return text, taste


def parts(root, cands: list[dict]) -> list[list[dict]]:
    """The candidates in parts that fit, broken between scans: one part, unless the book is so big
    (mostly a transcription that marks its text as headings) that one request sees too little of it."""
    size = len("\n".join(sheet(root, cands, TASTES[-1]))) / CHARS
    if size <= WHOLE:
        return [cands]
    n = int(size / (0.75 * BUDGET)) + 1   # with the printed contents and the words of the text, each part must stay under BUDGET
    out, cur = [], []
    for c in cands:
        if len(cur) >= len(cands) / n and c["scan"] != cur[-1]["scan"]:
            out.append(cur)
            cur = []
        cur.append(c)
    return out + [cur]


def answers(root, folder: Path, cands: list[dict], raw: Path) -> tuple[dict, float, int, int, int]:
    """The answer for the whole book, and its cost, tokens in and out, and the words shown."""
    whole = {"entries": [], "missing": [], "series": []}
    cost = tokens = spent = taste = 0
    pieces = parts(root, cands)
    for k, part in enumerate(pieces):
        text, taste = request(root, folder, part)
        if k:
            sofar = "\n".join(f"{'  ' * (int(e['level']) - 1)}{e['title']}" for e in whole["entries"][-CARRY:])
            text = (f"This is part {k + 1} of {len(pieces)} of the book. Go on with the same levels and series, "
                    f"the same labels too: {json.dumps(whole['series'], ensure_ascii=False)}\n"
                    f"The outline so far ends:\n{sofar}\n\n" + text)
        reply = ask(PROMPT, [{"type": "text", "text": text}], raw, text)
        cost, tokens, spent = cost + reply["cost"], tokens + reply["input"], spent + reply["output"]
        for name in ("entries", "missing"):
            whole[name] += reply["answer"].get(name, [])
        known = {str(s.get("label")) for s in whole["series"]}
        whole["series"] += [s for s in reply["answer"].get("series", []) if str(s.get("label")) not in known]
        whole.setdefault("book", reply["answer"].get("book"))
    return whole, cost, tokens, spent, taste


def ask(prompt: Path, content: list[dict], raw: Path, key: str) -> dict:
    """The command line's answer, kept under the hash of the prompt and key, as {"answer", "cost"...}."""
    system = prompt.read_text(encoding="utf-8")
    path = raw / f"{hashlib.sha256((system + chr(0) + key + MODEL + EFFORT).encode('utf-8')).hexdigest()[:20]}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if LIMITED.is_set():
        raise UsageLimit("usage limit reached earlier in this run")
    # common/via_claude.call: no API key and no CLAUDE* in its environment (the subscription, and not as part of a
    # Claude app this was started from); this command line, folder, timeout and the turn without a newline after it
    # are this script's own, as they were (REFACTOR_PLAN.md step 60)
    result, done = call(
        ["--model", MODEL, "--effort", EFFORT, "--system-prompt-file", str(prompt), "--tools", "",
         "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands", "--no-session-persistence",
         "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"],
        content, cli=str(CLAUDE), cwd=raw, timeout=TIMEOUT, encoding=None, newline=False)
    if not result or result.get("is_error"):
        if LIMIT_RE.search(f"{(result or {}).get('result')} {done.stderr[-300:]}"):
            LIMITED.set()
            raise UsageLimit(f"claude: {(result or {}).get('result') or done.stderr[-300:]}")
        raise RuntimeError(f"claude {done.returncode}: {(result or {}).get('result') or done.stdout[-300:]} {done.stderr[-300:]}")
    body = result.get("result") or ""
    u = result.get("usage", {})
    kept = {"answer": json.loads(body[body.index("{"):body.rindex("}") + 1]), "cost": result.get("total_cost_usd") or 0,
            "input": (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0),
            "output": u.get("output_tokens") or 0, "seconds": (result.get("duration_ms") or 0) // 1000}
    path.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    return kept


def numbers(text: str) -> set[str]:
    return {n.lower() for n in NUMBER.findall(text)}


def labelled(text: str) -> set[str]:
    """The numbers of a text that stand behind a label (CAP. XVII), not in a citation or date."""
    return {n.lower() for n in LABELLED.findall(text)}


def off_custos(ids: list[int], by_id: dict, cands: list[dict]) -> list[int]:
    """The ids of a heading without the catchword at the foot of the page before it: a page's last
    line whose letters the first line of the next page begins with. The heading then starts on the
    next page, at the lines the model gave there or else at that first line."""
    first = by_id[ids[0]]
    if not first.get("last") or first["level"] == -2:
        return ids
    nxt = next((c for c in cands if c["scan"] > first["scan"] and c["level"] != -2), None)
    word = T.folded(first["text"])
    if not (nxt and nxt["scan"] == first["scan"] + 1 and word and T.folded(nxt["text"]).startswith(word)):
        return ids
    return [i for i in ids if by_id[i]["scan"] > first["scan"]] or [nxt["id"]]


def checked(answer: dict, cands: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """The entries that hold, those that are to be looked for on the scans, and what was done.
    A title that does not resemble its lines can still hold when the lines carry its number. Else
    the line it came from may stand nearby; if not, the scan must tell."""
    by_id = {c["id"]: c for c in cands}
    notes, look = [], []
    entries = []
    for e in answer["entries"]:
        known = [i for i in e["ids"] if i in by_id]
        if not known:
            notes.append(f"no line: {e['title']!r}")
            look.append({"title": e["title"], "level": e["level"], "scan": None, "after": entries[-1]["ids"][-1] if entries else None})
            continue
        moved = off_custos(known, by_id, cands)
        if moved != known:
            notes.append(f"{e['title']!r} off the catchword {by_id[known[0]]['text']!r}")
            known = moved
        heads = [i for i in known if by_id[i]["level"] == -2]
        lines = [i for i in known if by_id[i]["level"] != -2]
        said = " ".join(e["title"].split())
        text = lambda ids: " ".join(by_id[i]["text"] for i in ids)
        if heads and lines and T.like(said, text(lines)) < T.MIN_LIKE and T.like(said, text(heads)) >= T.MIN_LIKE:
            known = heads   # check() would keep the lines, which are not what the title came from
        src = text(known)
        e = {**e, "ids": known}
        if T.like(said, src) < T.MIN_LIKE and not (len(T.folded(src)) >= 4 and T.like(src, said) >= 0.8):
            common = numbers(said) & labelled(src)
            near = [c for c in cands if abs(c["scan"] - by_id[known[0]]["scan"]) <= NEARBY and T.like(said, c["text"]) >= T.MIN_LIKE]
            if common:
                e["said"], e["title"] = said, src   # for check(), which then gives the title back
                notes.append(f"title {said!r} kept on the number of {src!r}")
            elif near:
                e["ids"] = [near[0]["id"]]
                notes.append(f"title {said!r} moved from {src!r} to {near[0]['text']!r}")
            else:
                notes.append(f"title {said!r} is not in {src!r}: to the scan")
                look.append({"title": said, "level": e["level"], "scan": by_id[known[0]]["scan"], "after": entries[-1]["ids"][-1] if entries else None})
                continue
        entries.append(e)
    good, bad = T.check(entries, cands)
    for e in good:
        if "said" in e:
            e["title"] = e.pop("said")
    return good, look, notes + bad


def looked(wanted: list[dict], good: list[dict], cands: list[dict], scans, raw: Path, title: str) -> tuple[list[dict], list[str], float]:
    """Each wanted division sought on the few scans where it must stand: new entries, with no line
    to mark, as [{scan, line, lines, level, title}], what was done, and the cost."""
    if scans.doc is None:
        return [], [f"no scans for {len(wanted)} wanted"] if wanted else [], 0.0
    by_id = {c["id"]: c for c in cands}
    starts = sorted(by_id[e["ids"][0]]["scan"] for e in good)
    found, notes, cost = [], [], 0.0
    for w in wanted[:LOOK_MOST]:
        if w.get("scan") is not None:
            lo = hi = w["scan"]
        elif w.get("after") in by_id:
            lo = by_id[w["after"]]["scan"]
            k = bisect.bisect_right(starts, lo)
            hi = starts[k] if k < len(starts) else lo + LOOK_SCANS
        else:
            notes.append(f"no place to look for {w['title']!r}")
            continue
        pages = list(range(lo, min(hi, lo + LOOK_SCANS - 1) + 1))
        content = [{"type": "text", "text": f"Book: {title}\nThe division sought: {w['title']}\n"}]
        for n in pages:
            image = scans.image(n, px=LOOK_PX)
            if image:
                content += [{"type": "text", "text": f"Scan {n}:"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image).decode()}}]
        reply = ask(SCAN_PROMPT, content, raw, f"{title}\0{w['title']}\0{pages}")
        cost += reply["cost"]
        a = reply["answer"]
        if a.get("found") and a.get("scan") in pages and a.get("title"):
            found.append({"scan": a["scan"], "line": 0, "lines": [], "level": min(T.LEVELS, max(1, int(w["level"] or 1))), "title": " ".join(a["title"].split())})
            notes.append(f"{w['title']!r} found on scan {a['scan']}")
        else:
            notes.append(f"{w['title']!r} not on scans {pages[0]}-{pages[-1]}")
    return found, notes, cost


def tally(answer: dict, toc: list[dict]) -> list[dict]:
    """Each series the model promised, and how many members of it stand in the contents: the entries
    that begin with its label, or the entries of its level if those are more. Series of one label
    count together, with the promise as their sum."""
    series = {}
    for s in answer.get("series", []):
        key = (str(s.get("label") or "").strip(), s.get("level"))
        series.setdefault(key, {**s, "count": 0})["count"] += int(s.get("count") or 0)
    out = []
    for (label, level), s in series.items():
        start = T.folded(label)[:5]
        named = [e for e in toc if start and T.folded(e["title"]).startswith(start)]
        kin = [e for e in toc if e["level"] == level]
        out.append({**s, "found": len(max(named, kin, key=len))})
    return out


def doubted(series: list[dict], toc: list[dict], scans: int) -> str:
    """Why this outline should not be used, or "": it falls short of its own promise, or leaves a
    great part of the book without an entry."""
    if not toc:
        return "no entries"
    for s in series:
        if s["count"] >= SHORT_MIN and s["found"] < SHORT * s["count"]:
            return f"{s['found']} of {s['count']} of series {s.get('label')!r}"
    at = [0] + sorted(e["scan"] for e in toc) + [scans]
    empty = max(b - a for a, b in zip(at, at[1:]))
    if scans >= EMPTY_MIN and empty > EMPTY * scans:
        return f"{empty} of {scans} scans without an entry"
    return ""


def toc(root) -> list[dict]:
    """The entries of data/<work>/toc_context.json of this book, or [] if it has none or a doubt."""
    source = root.get("source")
    path = T.DATA / Path(source).stem / OUT_NAME if source else None
    if path is None or not path.is_file():
        return []
    made = json.loads(path.read_text(encoding="utf-8"))
    return [] if made.get("doubt") else made["toc"]


def run(work: str, out: Path | None = None, only_sheet: bool = False, redo: bool = False) -> dict | None:
    folder = T.folder_of(work)
    if not (redo or only_sheet) and (out / f"{work}.json" if out else folder / OUT_NAME).is_file():
        return None   # done before: --redo to make it again (the answers kept in raw cost nothing)
    xml = folder / "book.xml"
    root = etree.parse(str(xml), etree.XMLParser(recover=True)).getroot()
    cands = T.candidates(root)
    if only_sheet:
        text, taste = request(root, folder, cands)
        (out or folder).joinpath(f"{work}.txt").write_text(text, encoding="utf-8")
        print(f"{work}: {len(cands)} candidates, {len(text)} characters, {taste} words", flush=True)
        return None
    raw = out / "raw" if out else folder / RAW
    raw.mkdir(parents=True, exist_ok=True)
    answer, first, tokens, spent, taste = answers(root, folder, cands, raw)
    good, look, notes = checked(answer, cands)
    wanted = look + [{"title": m.get("title", ""), "level": m.get("level"), "scan": None, "after": m.get("after")} for m in answer.get("missing", [])]
    scans = T.Scans(folder, xml)
    found, seen, cost = looked(wanted, good, cands, scans, raw, root.get("source", folder.name))
    scans.close()
    toc = sorted(T.outline(good, cands) + found, key=lambda e: e["scan"])
    series = tally(answer, toc)
    why = doubted(series, toc, sum(1 for _ in root.iter("page")))
    result = {"work": folder.name, "made": dt.datetime.now().isoformat(timespec="seconds"), "model": MODEL,
              "prompt": hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:16], "candidates": len(cands), "cost_usd": round(first + cost, 5),
              "dropped": notes + seen, "toc": toc, "book": answer.get("book"), "series": series, "taste": taste, "doubt": why,
              "input": tokens, "output": spent}
    (out / f"{work}.json" if out else folder / OUT_NAME).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{work}: {len(toc)} entries ({len(found)} from scans of {len(wanted)} wanted), {taste} words, {tokens} in, "
          f"${result['cost_usd']:.3f}{', DOUBT ' + why if why else ''}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="work ids (W03402)")
    ap.add_argument("--list", type=Path, help="a file with a work id per line")
    ap.add_argument("--out", type=Path, help="write the answers and results here, not with the book")
    ap.add_argument("--sheet", action="store_true", help="only write the sheet, ask nothing")
    ap.add_argument("--parallel", type=int, default=3, help="books at once")
    ap.add_argument("--force", "--redo", dest="redo", action="store_true", help="also the books that have their outline already")
    args = ap.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    works = args.works + (args.list.read_text().split() if args.list else [])

    def one(w):
        try:
            return run(w, args.out, args.sheet, args.redo)
        except UsageLimit as e:
            print(f"{w}: stopped, {e}", flush=True)
        except Exception as e:   # one book's trouble is no reason to stop the others
            print(f"{w}: {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, works))


if __name__ == "__main__":
    main()
