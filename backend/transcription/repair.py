"""Transcribe the pages that audit.py found missing, work by work, without repeating work.

This does not call the API itself: it runs transcribe.py for one work at a time (so every page
goes through the normal validation, book.xml and book.json are rebuilt, and a re-run resumes),
and it keeps a progress file so an interrupted run can be picked up where it stopped.

  python transcription/repair.py --category pilot+gaps          # what audit.py finds missing
  python transcription/repair.py --category pilot --limit 1 --dry-run   # one work, print the command
  python transcription/repair.py --category pilot --restart-on "session limit"

Default: --model claude-haiku-5-5 --batch (batch halves the price; transcribe.py polls until the
batch is done). Pages without an OCR text layer are transcribed too; transcribe.py lists them
in no_ocr.txt.
Log of every run: logs/repair_<timestamp>.log; progress: logs/repair_progress.tsv
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/
from common.llm import MODEL_HAIKU  # noqa: E402
from common.pagespec import page_numbers  # noqa: E402
from common.paths import DATA, LOGS, PDF_DIR, refuse_foreign  # noqa: E402
from common.works import ids_of  # noqa: E402

HERE = Path(__file__).resolve().parent


def audit_rows(category: str) -> list[tuple[str, str]]:
    """(work, pages) for every category in `category` (one category, or several separated by
    commas: a work in two of them is run once, with the pages of all of them)."""
    seen: dict[str, set[int]] = {}
    order: list[str] = []
    for cat in category.split(","):
        cmd = [sys.executable, str(HERE / "audit.py"), "--list", cat]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            work, _, pages = line.partition("\t")
            nums = set(page_numbers(pages, skip_empty=True))
            if not nums:
                continue
            if work not in seen:
                seen[work], order = set(), order + [work]
            seen[work] |= nums
    return [(w, ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in _runs(sorted(seen[w])))) for w in order]


def _runs(nums: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", default="pilot,gaps,failed",
                    help="audit.py categories; a comma lists several (all of them are run) "
                         "(default: pilot,gaps,failed)")
    ap.add_argument("--model", default=MODEL_HAIKU)
    ap.add_argument("--no-batch", action="store_true", help="sync mode (twice the price, no batch wait)")
    ap.add_argument("--limit", type=int, help="stop after this many works")
    ap.add_argument("--min-pages", type=int, default=1, help="skip works with fewer pages than this")
    ap.add_argument("--only", help="only these work ids, comma-separated (W05482,W05533)")
    ap.add_argument("--restart-on", action="append", default=[],
                    help="start the work again when the log of its run contains this text (e.g. "
                         "'session limit', 'invalid_request_error'); the progress file then no longer "
                         "counts that run as done")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--all-works", action="store_true",
                    help="also run the works with no page to transcribe (nothing would change)")
    args = ap.parse_args()
    refuse_foreign(DATA)

    LOGS.mkdir(exist_ok=True)
    progress = LOGS / "repair_progress.tsv"
    todo = audit_rows(args.category)
    if args.only:
        want = ids_of(args.only.split(","))
        todo = [(w, p) for w, p in todo if w.split("_")[0] in want]
    todo = [(w, p) for w, p in todo if int(p.count(",")) + 1 >= args.min_pages]
    # A work with no page to transcribe is skipped: running transcribe.py on it would only
    # overwrite the pages that are already there.
    total = len(todo)
    if not args.all_works:
        todo = [(w, p) for w, p in todo if p]
    print(f"{len(todo)} work(s) to run, {total - len(todo)} skipped (no page to transcribe), "
          f"category {args.category}", flush=True)
    log = LOGS / f"repair_{dt.datetime.now():%Y-%m-%d_%H%M%S}.log"
    for i, (work, pages) in enumerate(todo, 1):
        pdf = PDF_DIR / f"{work}.pdf"
        if not pdf.exists():
            print(f"[{i}/{len(todo)}] {work}: no PDF, skipped", flush=True)
            continue
        cmd = [sys.executable, str(HERE / "transcribe.py"), str(pdf), "--pages", pages,
               "--model", args.model] + ([] if args.no_batch else ["--batch"])
        print(f"[{i}/{len(todo)}] {work}: {pages.split(',')[0]}...{pages.split(',')[-1]} "
              f"({pages.count(',') + 1} page(s))", flush=True)
        if args.dry_run:
            print("  " + " ".join(cmd), flush=True)
            continue
        out_root = DATA / work
        before = _transcribed(out_root)
        run = subprocess.run(cmd, capture_output=True, text=True)
        text = run.stdout + run.stderr
        (LOGS / f"repair_{work}.log").write_text(text, encoding="utf-8")
        # The work log of this run is what --restart-on looks at.
        restart = any(kind in text for kind in args.restart_on)
        after = _transcribed(out_root)
        new = len(after - before)
        status = "restart" if restart else ("empty" if new == 0 else "done")
        with log.open("a", encoding="utf-8") as f:
            f.write(f"{status}\t{work}\t{len(after)}\t{run.returncode}\n")
        tail = [l for l in text.splitlines() if l.startswith("Spent this run")]
        print(f"  {'FAILED' if run.returncode else 'ok'} {tail[-1] if tail else ''} "
              f"(+{new} page file(s))" + (" -> restart later" if restart else ""), flush=True)
        if restart:
            sys.exit(f"{work}: the run hit '{args.restart_on}'; stopped. Start it again to resume.")
        if args.limit and i >= args.limit:
            print("Stopped at --limit", flush=True)
            break


def _transcribed(root: Path) -> set[str]:
    """The pages of a work that hold text (a missing-page mark is not a transcription)."""
    d = root / "pages"
    if not d.exists():
        return set()
    return {f.name for f in d.glob("p*.xml")
            if not f.name.endswith(".para.xml")
            and not f.read_text(encoding="utf-8", errors="ignore").lstrip().startswith('<page type="missing')}


if __name__ == "__main__":
    main()
