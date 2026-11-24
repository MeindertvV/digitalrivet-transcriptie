"""Transcribe with Haiku batches until a dollar budget is used up.

Phase 1: the pages audit.py lists as missing (pilot, gaps, page-1, failed) in works already begun.
Phase 2: whole untranscribed works (no data/<work>/book.json), in random order.

Runs transcribe.py (Haiku default, --batch) for several works at a time. Spend is the sum of
cost_usd in the raw/ files each run wrote; before a work starts, its pages x RESERVE are held
back from the budget, so the running works together can never pass it.

Phase 2 skips the works in special_layout.tsv (a layout the prompt is not made for) and adds the works
whose layout_check.py verdict is "columns" or "marginalia" to it. Once the API refuses for lack of credit, no new work starts.
The number of works at a time can be changed while running, in logs/budget_run_<date>/parallel.txt.
--special runs the works of special_layout.tsv instead (smallest first, or those given with
--works): each with its layout standard (profile.py, transcribe.py --profile).

  python transcription/budget_run.py --budget 100 --parallel 8 [--dry-run]
  python transcription/budget_run.py --budget 5 --special --no-phase1 [--works W00339,W07119]
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pymupdf

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))  # backend/: common/ and the other stages
from common import dubbel  # noqa: E402
from common.pagespec import page_numbers  # noqa: E402
from common.paths import DATA, LOGS, PDF_DIR, refuse_foreign  # noqa: E402
from common.works import ids_of  # noqa: E402
from transcription import layout_check  # noqa: E402

TRANSCRIBE = BACKEND / "transcription" / "transcribe.py"
AUDIT = BACKEND / "transcription" / "audit.py"
SPECIAL = dubbel.SPECIAL   # works with a layout of their own, kept in git: the live one, as the gate reads it
RESERVE = 0.0015       # $ held back per page while a work runs (measured mean is $0.0007)
FRESH_SECONDS = 1800   # skip PDFs written in the last half hour: the downloader may still be on them
OUT_OF_CREDIT = threading.Event()   # set when the API refuses for lack of credit: start nothing new


def n_pages(spec: str) -> int:
    return len(page_numbers(spec, skip_empty=True))


def phase1() -> list[tuple[str, str | None, int]]:
    seen: dict[str, str] = {}
    for cat in ("pilot", "gaps", "page-1", "failed"):
        out = subprocess.run([sys.executable, str(AUDIT), "--list", cat], capture_output=True, text=True,
                             encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                             check=True).stdout
        for line in out.splitlines():
            work, _, pages = line.partition("\t")
            if pages:
                seen[work] = ",".join(x for x in (seen.get(work, ""), pages) if x)
    return [(w, p, n_pages(p)) for w, p in seen.items()]


def special_layout() -> set[str]:
    """The work ids listed in special_layout.tsv."""
    if not SPECIAL.exists():
        return set()
    return {l.split("\t")[0] for l in SPECIAL.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")}


def phase2(seed: int) -> list[tuple[str, str | None, int]]:
    now = time.time()
    pdfs = [p for p in PDF_DIR.glob("*.pdf")
            if not (DATA / p.stem / "book.json").exists() and now - p.stat().st_mtime > FRESH_SECONDS]
    random.Random(seed).shuffle(pdfs)
    special = special_layout()
    for p in pdfs:
        if p.stem[:6] in special:
            continue
        try:
            n = pymupdf.open(p).page_count
            layout, vals = layout_check.verdict(p.stem)
        except Exception:
            continue
        # Columns, tables or heavy marginal notes: not for this run.
        if layout:
            with SPECIAL.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"{p.stem[:6]}\t{layout}\t{n}\t{p.stem[7:]}\n")
            continue
        yield (p.stem, None, n)


def special_works(only: set[str] | None) -> list[tuple[str, str | None, int]]:
    """The works of special_layout.tsv without book.json, smallest first (or only those in only)."""
    out = []
    for line in SPECIAL.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        wid = line.split("\t")[0]
        if only and wid not in only:
            continue
        pdf = next(iter(sorted(PDF_DIR.glob(f"{wid}_*.pdf"))), None)
        if pdf is None or (not only and (DATA / pdf.stem / "book.json").exists()):
            continue
        try:
            out.append((pdf.stem, None, pymupdf.open(pdf).page_count))
        except Exception:
            continue
    return sorted(out, key=lambda x: x[2])


def not_held(item: tuple[str, str | None, int]) -> bool:
    """False for a work without book.json that waits as a probable duplicate (common/dubbel.py, DUBBELEN.md):
    transcribe.py would refuse it anyway. Repairs (phase 1) of finished works always go on."""
    if (DATA / item[0] / "book.json").exists() or not (why := dubbel.held(item[0])):
        return True
    print(f"wacht: {why}", flush=True)
    return False


def raw_spend(work: str, since: float) -> float:
    total = 0.0
    for f in (DATA / work / "raw").glob("p*.json"):
        if f.stat().st_mtime >= since:
            try:
                total += json.loads(f.read_text(encoding="utf-8")).get("cost_usd", 0)
            except Exception:
                pass
    return total


def run(work: str, pages: str | None, log_dir: Path, profile: bool = False) -> tuple[float, int]:
    cmd = [sys.executable, str(TRANSCRIBE), str(PDF_DIR / f"{work}.pdf"), "--batch", "--poll", "120"]
    if profile:
        cmd.append("--profile")
    if pages:
        cmd += ["--pages", pages]
    start = time.time() - 5
    with (log_dir / f"{work}.log").open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=BACKEND,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    text = (log_dir / f"{work}.log").read_text(encoding="utf-8", errors="ignore").lower()
    if "credit balance" in text or "billing" in text:
        OUT_OF_CREDIT.set()
    return raw_spend(work, start), rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--spent-before", type=float, default=0.0,
                    help="$ an earlier, interrupted run of this budget already spent")
    ap.add_argument("--first", help="comma-separated works to run before everything else")
    ap.add_argument("--exclude", help="comma-separated works not to start (e.g. still running elsewhere)")
    ap.add_argument("--no-phase1", action="store_true", help="skip the audit; only random works")
    ap.add_argument("--special", action="store_true",
                    help="the works of special_layout.tsv, each with its layout standard (transcribe.py --profile)")
    ap.add_argument("--only", "--works", dest="works", help="with --special: only these work ids, comma-separated")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    refuse_foreign(DATA)
    dubbel._state()   # read the gate now: a broken list of decisions stops here, before any work is started

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    log_dir = LOGS / f"budget_run_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = log_dir / "summary.tsv"

    p1 = [] if args.no_phase1 else phase1()
    print(f"phase 1: {len(p1)} work(s), {sum(n for *_, n in p1)} page(s)", flush=True)
    # Works an interrupted run left half done go first: their submitted batches are paid for anyway.
    first = [(w, None, pymupdf.open(PDF_DIR / f"{w}.pdf").page_count)
             for w in (args.first.split(",") if args.first else []) if w not in {x for x, *_ in p1}]
    skip = {w for w, *_ in first + p1} | set(args.exclude.split(",") if args.exclude else [])
    later = special_works(ids_of(args.works.split(",")) if args.works else None) if args.special else phase2(args.seed)
    queue = itertools.chain(first, p1, (x for x in later if x[0] not in skip))
    queue = (x for x in queue if not_held(x))   # probable duplicates wait for a decision
    if args.dry_run:
        for i, (w, p, n) in enumerate(queue):
            if i >= len(p1) + 10:
                break
            print(f"{'P1' if p else 'P2'}\t{n}\t{w}\t{(p or '')[:60]}")
        return

    # The number of works at a time can change while running: write it to this file.
    knob = log_dir / "parallel.txt"
    knob.write_text(str(args.parallel))

    def parallel() -> int:
        try:
            return max(1, int(knob.read_text().strip()))
        except Exception:
            return args.parallel

    spent, held, running = args.spent_before, 0.0, {}
    with ThreadPoolExecutor(64) as pool:
        nxt = next(queue, None)
        while nxt or running:
            if OUT_OF_CREDIT.is_set() and nxt:
                print("API credit used up: starting no new works", flush=True)
                nxt = None
            while nxt and len(running) < parallel():
                work, pages, n = nxt
                cost = n * RESERVE
                if spent + held + cost > args.budget:
                    if not running:   # even alone it does not fit: try a smaller work
                        nxt = next(queue, None)
                        continue
                    break
                held += cost
                running[pool.submit(run, work, pages, log_dir, args.special)] = (work, n, cost)
                print(f"start {work} ({n} p, held ${cost:.2f}; spent ${spent:.2f}, held ${held:.2f})", flush=True)
                nxt = next(queue, None)
            if not running:
                break
            done, _ = wait(running, timeout=60, return_when=FIRST_COMPLETED)
            for f in done:
                work, n, cost = running.pop(f)
                held -= cost
                try:
                    used, rc = f.result()
                except Exception as e:
                    used, rc = 0.0, f"error {e}"
                spent += used
                with summary.open("a", encoding="utf-8") as fh:
                    fh.write(f"{work}\t{n}\t{used:.4f}\t{rc}\t{spent:.4f}\n")
                print(f"done  {work}: ${used:.3f} rc={rc}; total ${spent:.2f}", flush=True)
    print(f"Budget run finished: ${spent:.2f} of ${args.budget:.2f}", flush=True)


if __name__ == "__main__":
    main()
