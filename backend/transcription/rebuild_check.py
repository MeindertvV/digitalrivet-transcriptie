"""What would change if book.xml were made again from the saved pages: a look only, nothing in data/ is written.

Each work is rebuilt in a temporary folder, twice, and compared page by page:

  merge    from pages/pNNNN.xml as they are (transcribe.py --merge-only), compared with today's book.xml.
           A difference here is the merge's own new rules (the model's paragraph marks since 5573ec1), or an edit
           made in book.xml and not in the pages: that edit a merge would undo (P19).
  resplit  from the model's pages pNNNN.para.xml, cut into lines again with the OCR layer (transcribe.py --resplit),
           compared with the merge above. A difference here is split_lines' own new rules, or an edit made in
           pNNNN.xml and not in pNNNN.para.xml: that edit a resplit would undo.

Per page the difference is one kind, the worst that applies:
  same     nothing but whitespace
  para     only where paragraphs start (para="true")
  other    the lines' text is the same, something else differs (a note's place, a catchword, a warning, an attribute)
  hyphen   the lines differ only in a hyphen at the end (split_lines puts back the one the OCR did not see)
  lines    the text of the lines differs otherwise, or their number: the translation has to be linked again
  added / gone   the page is only in the rebuild / only in the book.xml compared with

One line per work goes to logs/rebuild_check/<name>.jsonl, with up to EXAMPLES differences of each kind to look at.
A work already in that file is skipped, so a run stopped by --until goes on where it was the next time.
CPU only, no model and no cost; the resplit opens each PDF, so the whole library is a night's work (night job
"herbouwcontrole").

Examples
  python transcription/rebuild_check.py W03335 W01206           # these works, to the screen and the log
  python transcription/rebuild_check.py --all --until 06:30      # every transcribed work, until half past six
  python transcription/rebuild_check.py --report                 # the counts of the log so far
  python transcription/rebuild_check.py --report --list lines    # the works with a page of that kind
"""

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.paths import DATA, LOGS, PDF_DIR  # noqa: E402
from common.works import ids_of  # noqa: E402

LOG_DIR = LOGS / "rebuild_check"
KINDS = ("same", "para", "other", "hyphen", "lines", "added", "gone")   # from least to most
EXAMPLES = 5


# ---------------------------------------------------------------- comparing two books

def norm(s: str | None) -> str:
    return " ".join((s or "").split())


def line_texts(page: ET.Element) -> list[str]:
    main = page.find("main_text")
    return [norm("".join(l.itertext())) for l in main if l.tag == "l"] if main is not None else []


def shape(el: ET.Element, drop_para: bool) -> tuple:
    """The element as a comparable value: tag, attributes, text and children, whitespace made equal."""
    attrs = {k: v for k, v in el.attrib.items() if not (drop_para and k == "para")}
    return (el.tag, tuple(sorted(attrs.items())), norm(el.text), norm(el.tail),
            tuple(shape(c, drop_para) for c in el))


def first_difference(a: ET.Element, b: ET.Element, path: str = "") -> str:
    """Where two pages first differ, in a few words: the path to the element and what differs."""
    here = f"{path}/{a.tag}" if path else a.tag
    if a.tag != b.tag:
        return f"{here}: <{a.tag}> / <{b.tag}>"
    if a.attrib != b.attrib:
        keys = sorted(k for k in set(a.attrib) | set(b.attrib) if a.get(k) != b.get(k))
        return f"{here}@{keys[0]}: {a.get(keys[0])!r} / {b.get(keys[0])!r}"
    if norm(a.text) != norm(b.text):
        return f"{here} text: {norm(a.text)[:60]!r} / {norm(b.text)[:60]!r}"
    for i, (x, y) in enumerate(zip(a, b)):
        if shape(x, False) != shape(y, False):
            if x.tag == "l" and y.tag == "l":
                here_l = f"{here}/l[{x.get('n') or i + 1}]"
                if norm("".join(x.itertext())) != norm("".join(y.itertext())):
                    return f"{here_l}: {norm(''.join(x.itertext()))[:60]!r} / {norm(''.join(y.itertext()))[:60]!r}"
            return first_difference(x, y, here)
    if len(a) != len(b):
        more, other = (a, "rebuild lacks") if len(a) > len(b) else (b, "rebuild adds")
        return f"{here}: {other} <{more[min(len(a), len(b))].tag}>"
    return f"{here}: tail differs"


def unhyphened(lines: list[str]) -> list[str]:
    return [l.rstrip("-=¬") for l in lines]


def line_difference(old: ET.Element, new: ET.Element) -> str:
    """The first line whose text differs, from just before where it starts to differ; or the number of lines."""
    a, b = line_texts(old), line_texts(new)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            k = next((j for j, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
            k = max(0, k - 25)
            return f"l{i + 1}: {x[k:k + 60]!r} / {y[k:k + 60]!r}"
    return f"{len(a)} lines / {len(b)} lines" + (f", first extra: {(a + b)[min(len(a), len(b))][:60]!r}")


def page_kind(old: ET.Element, new: ET.Element) -> str:
    if shape(old, False) == shape(new, False):
        return "same"
    a, b = line_texts(old), line_texts(new)
    if a != b:
        return "lines" if unhyphened(a) != unhyphened(b) else "hyphen"
    if shape(old, True) == shape(new, True):
        return "para"
    return "other"


def compare(old: ET.Element, new: ET.Element) -> tuple[Counter, dict]:
    """Per kind the number of pages, and up to EXAMPLES of each kind but same and para: (scan, where it differs)."""
    a = {int(p.get("n")): p for p in old.iter("page")}
    b = {int(p.get("n")): p for p in new.iter("page")}
    counts, examples = Counter(), {}
    for n in sorted(set(a) | set(b)):
        if n not in b:
            kind, where = "gone", ""
        elif n not in a:
            kind, where = "added", ""
        else:
            kind = page_kind(a[n], b[n])
            where = (line_difference(a[n], b[n]) if kind in ("lines", "hyphen")
                     else first_difference(a[n], b[n]) if kind == "other" else "")
        counts[kind] += 1
        if kind not in ("same", "para", "hyphen") and len(examples.setdefault(kind, [])) < EXAMPLES:
            examples[kind].append([n, where])
    return counts, examples


# ---------------------------------------------------------------- rebuilding in a temporary folder

def rebuild(work_dir: Path, tmp: Path, resplit: bool, source_name: str) -> ET.Element:
    """book.xml as transcribe.py --merge-only (resplit False) or --resplit would make it, in tmp."""
    from transcription.page_store import Output   # here: a worker loads it once, the report never
    pages = tmp / "pages"
    pages.mkdir(parents=True)
    for f in (work_dir / "pages").glob("p*.xml"):
        shutil.copy2(f, pages / f.name)
    if (work_dir / "raw").is_dir():
        (tmp / "raw").symlink_to((work_dir / "raw").resolve())   # merge only reads it (the model of each page)
    if resplit:
        from transcription.pdf_source import Book
        book = Book(PDF_DIR / f"{work_dir.name}.pdf")
        out = Output(tmp, book)
        out.resplit()
    else:
        book = SimpleNamespace(source=Path(source_name))
        out = Output(tmp, book)
    out.merge(book)
    return ET.parse(tmp / "book.xml").getroot()


def check(work_dir: Path) -> dict:
    """The comparison of one work, as the line the log gets."""
    t0 = time.time()
    row = {"work": work_dir.name, "at": datetime.now().isoformat(timespec="seconds")}
    try:
        old = ET.parse(work_dir / "book.xml").getroot()
        row["pages"] = sum(1 for _ in old.iter("page"))
        row["para_pages"] = sum(1 for _ in (work_dir / "pages").glob("p*.para.xml"))
        with tempfile.TemporaryDirectory(prefix="rebuild_check_") as t, contextlib.redirect_stdout(io.StringIO()):
            merged = rebuild(work_dir, Path(t) / "merge", False, old.get("source") or f"{work_dir.name}.pdf")
            counts, examples = compare(old, merged)
            row["merge"], row["merge_examples"] = dict(counts), examples
            if (PDF_DIR / f"{work_dir.name}.pdf").is_file():
                split = rebuild(work_dir, Path(t) / "resplit", True, "")
                counts, examples = compare(merged, split)
                row["resplit"], row["resplit_examples"] = dict(counts), examples
            else:
                row["resplit_skipped"] = "geen PDF"
    except (Exception, SystemExit) as e:   # Book() exits on a PDF without pages
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    row["seconds"] = round(time.time() - t0, 1)
    return row


# ---------------------------------------------------------------- the run and the report

def works(names: list[str], every: bool) -> list[Path]:
    if every:
        return sorted(p.parent for p in DATA.glob("W*/book.xml") if (p.parent / "pages").is_dir())
    found = []
    for w in sorted(ids_of(names)):
        dirs = sorted(d for d in DATA.glob(f"{w}_*") if (d / "book.xml").is_file())
        if not dirs:
            print(f"{w}: geen werkmap met book.xml", file=sys.stderr)
        found += dirs[:1]
    return found


def done_in(log: Path) -> set[str]:
    if not log.is_file():
        return set()
    done = set()
    for l in log.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(l)["work"])
        except (ValueError, KeyError, TypeError):   # empty, or cut off when a run was stopped: done again
            pass
    return done


def line_of(row: dict) -> str:
    if "error" in row:
        return f"{row['work'][:6]}  FOUT {row['error']}"
    def part(key):
        c = row.get(key)
        return " ".join(f"{k} {c[k]}" for k in KINDS if c and c.get(k)) if c else row.get(f"{key}_skipped", "-")
    return f"{row['work'][:6]}  {row['pages']:>4} p.  merge: {part('merge'):<34} resplit: {part('resplit')}  ({row['seconds']} s)"


def report(log: Path, kind: str | None) -> None:
    rows = []
    for l in (log.read_text(encoding="utf-8").splitlines() if log.is_file() else []):
        with contextlib.suppress(ValueError):   # a line cut off when a run was stopped
            rows.append(json.loads(l))
    ok = [r for r in rows if "error" not in r]
    print(f"{log}: {len(rows)} werken, {len(rows) - len(ok)} met een fout")
    for key in ("merge", "resplit"):
        pages, worst = Counter(), Counter()
        for r in ok:
            c = r.get(key) or {}
            pages.update(c)
            if c:
                worst[max(c, key=KINDS.index)] += 1
        print(f"\n{key}: pagina's per soort, en werken naar hun ergste pagina")
        for k in KINDS:
            print(f"  {k:<6} {pages[k]:>8} pagina's   {worst[k]:>5} werken")
        if key == "resplit":
            print(f"  zonder PDF: {sum(1 for r in ok if 'resplit_skipped' in r)} werken")
    if kind:
        print(f"\nwerken met een pagina '{kind}':")
        for r in ok:
            for key in ("merge", "resplit"):
                if (r.get(key) or {}).get(kind):
                    ex = (r.get(f"{key}_examples") or {}).get(kind, [])
                    print(f"  {r['work'][:6]} {key:<7} {r[key][kind]:>4}  " + "; ".join(f"p{n} {w}" for n, w in ex[:2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="work ids (W03335) or folder names")
    ap.add_argument("--all", action="store_true", help="every work with a book.xml and pages/")
    ap.add_argument("--log", default="library", help="name of the log in logs/rebuild_check/ (default: library)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--until", metavar="UU:MM", help="start no new work after this time (or seconds since 1970)")
    ap.add_argument("--again", action="store_true", help="also the works already in the log (a second line each)")
    ap.add_argument("--report", action="store_true", help="only count what the log holds")
    ap.add_argument("--list", metavar="KIND", choices=KINDS, help="with --report: the works with a page of this kind")
    args = ap.parse_args()
    log = LOG_DIR / f"{args.log}.jsonl"
    if args.report:
        report(log, args.list)
        return
    if not args.works and not args.all:
        ap.error("geef werken, of --all")
    todo = works(args.works, args.all)
    if not args.again:
        seen = done_in(log)
        todo = [d for d in todo if d.name not in seen]
    stop = None
    if args.until and args.until.isdigit():   # seconds, as the night runner hands it on (UNTIL_EPOCH)
        stop = datetime.fromtimestamp(int(args.until))
    elif args.until:
        h, m = map(int, args.until.split(":"))
        stop = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if stop <= datetime.now():
            stop += timedelta(days=1)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} werken te doen -> {log}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool, log.open("a", encoding="utf-8") as out:
        it = iter(todo)
        running = {}
        def submit():
            if stop and datetime.now() >= stop:
                return False
            d = next(it, None)
            if d is None:
                return False
            running[pool.submit(check, d)] = d
            return True
        for _ in range(args.workers * 2):
            if not submit():
                break
        while running:
            f = next(as_completed(running))
            running.pop(f)
            row = f.result()
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(line_of(row), flush=True)
            submit()
    if stop and datetime.now() >= stop:
        left = len(todo) - len(done_in(log) & {d.name for d in todo})
        print(f"gestopt om {args.until}: {left} werken nog te doen")
        if left:
            sys.exit(75)   # not finished: the night runner's "not now, no failure"; the queue keeps the job


if __name__ == "__main__":
    main()
