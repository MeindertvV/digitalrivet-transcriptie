"""What is missing in the transcription of every work, without any API call.

A page that was never transcribed looks exactly like a page that was blank in the book: both are
simply absent from book.json. This script says which pages of each scanned book are really missing
(absent, or marked <page type="missing">), and which were transcribed without an OCR layer and so
are not yet split into printed lines (no_ocr.txt).

Categories (a work can be in several):
  no-pages        book.json exists but has no pages (every page failed)
  pilot           only a handful of pages from a --pages pilot; the rest of the book never ran
  page-1          only the first page(s) were transcribed and then the run stopped
  gaps            pages missing in the middle or at the end of a book that runs further
  failed          failed.txt lists pages with their reason

Writes logs/taak_overzicht.csv (one row per work) and prints a summary. Use --list <category> to
print the page numbers of a category, e.g. to feed them to transcribe.py.

  python transcription/audit.py                    # summary + CSV
  python transcription/audit.py --detail           # also print the gaps per work
  python transcription/audit.py --list gaps        # the page numbers to transcribe
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the stages
from common.paths import DATA, PDF_DIR  # noqa: E402
from common.works import folder_in, id_of  # noqa: E402

LOGS = Path(__file__).resolve().parents[1] / "logs"
OUT = LOGS / "taak_overzicht.csv"

# A book whose scans stop well before the end of the PDF is interrupted; page furniture
# (binding, blank leaves, a plate) rarely takes up more than a few dozen scans.
TAIL_SLACK = 60
# A work with at most this many pages that is missing the rest looks like a pilot.
PILOT_MAX = 30


def pdf_pages(path: Path) -> int | None:
    """Number of scans in the PDF."""
    try:
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001 - an unreadable PDF is reported, not fatal
        return None


def runs(nums: list[int]) -> list[tuple[int, int]]:
    """Consecutive numbers as (first, last) pairs."""
    out: list[tuple[int, int]] = []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def omit(words: list[tuple[int, int]], most: int = 4) -> str:
    """'920-1234' or '73, 666, 739, 744, ...' - the first few runs, the rest counted."""
    parts = [f"{a}" if a == b else f"{a}-{b}" for a, b in words[:most]]
    if len(words) > most:
        parts.append(f"... ({len(words)} runs)")
    return ", ".join(parts)


def read_failed(root: Path) -> dict[int, str]:
    path = root / "failed.txt"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        num, _, why = line.partition("\t")
        if num.isdigit():
            out[int(num)] = why.strip()
    return out


def read_no_ocr(root: Path) -> list[int]:
    path = root / "no_ocr.txt"
    if not path.exists():
        return []
    return [int(l) for l in path.read_text(encoding="utf-8").split() if l.isdigit()]


def load(path: Path) -> dict[int, dict]:
    """Pages by scan number from a book.json, or {} if there is none."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {p["n"]: p for p in data.get("pages", [])}


def analyse(root: Path) -> dict | None:
    book = root / "book.json"
    if not book.exists():
        return None  # no transcription at all: step ①, not a transcription problem
    tr = load(book)  # empty for a book.json without pages: every page failed
    pdf = PDF_DIR / f"{root.name}.pdf"
    total = pdf_pages(pdf) if pdf.exists() else None
    if not total:
        return {"work": root.name, "status": "geen pdf", "scans": 0, "transcribed": len(tr),
                "last_page": 0, "to_do": 0, "gaps": "", "no_ocr": 0,
                "no_ocr_pages": "", "failed": 0, "reason": "the PDF is missing from E:\\PRDL_pdfs"}
    # A page marked as missing (<page type="missing">: failed, or skipped by an older
    # transcribe.py for lack of an OCR layer) is tried again by the next run: it is to do.
    done = {n for n, p in tr.items() if p.get("type") != "missing"}
    to_do = [n for n in range(1, total + 1) if n not in done]
    # Pages transcribed without an OCR layer keep one line per paragraph (transcribe.py lists
    # them in no_ocr.txt); --resplit cuts them into lines once they have one.
    no_ocr = read_no_ocr(root)
    failed = read_failed(root)
    failed_todo = [n for n in failed if n not in done]
    present = sorted(done)
    cats = []
    if not present:
        cats.append("no-pages")
    elif max(present) < total - TAIL_SLACK:
        cats.append("pilot" if len(present) <= PILOT_MAX else "page-1")
        if max(present) <= 2 and len(present) <= 3:
            cats.append("page-1")
    elif to_do or failed_todo:
        cats.append("gaps")
    if failed_todo:
        cats.append("failed")
    cats = list(dict.fromkeys(cats))
    if not cats:
        return None
    reasons = sorted({re.sub(r"\s+", " ", why) for why in failed.values()})
    return {
        "work": root.name,
        "status": "+".join(cats),
        "scans": total,
        "transcribed": len(present),
        "last_page": max(present) if present else 0,
        "to_do": len(to_do),
        "gaps": omit(runs(to_do)) if to_do else "",
        "no_ocr": len(no_ocr),
        "no_ocr_pages": omit(runs(no_ocr)) if no_ocr else "",
        "failed": len(failed_todo),
        "reason": "; ".join(r[:70] for r in reasons[:2]),
        "pages": ",".join(map(str, sorted(set(to_do) | set(failed_todo)))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detail", action="store_true", help="print the gaps of every work")
    ap.add_argument("--list", metavar="CATEGORY", help="print work<TAB>pages to transcribe for one category")
    ap.add_argument("--only", "--work", dest="work", help="only this work directory")
    ap.add_argument("--status", help="only rows with this status (default: all")
    args = ap.parse_args()

    roots = sorted(p for p in DATA.iterdir() if p.is_dir())
    if args.work:
        name = (folder_in(DATA, id_of(args.work)) or Path(args.work)).name
        roots = [p for p in roots if p.name == name]
    rows = []
    for i, root in enumerate(roots, 1):
        row = analyse(root)
        if row:
            rows.append(row)
        if i % 250 == 0:
            print(f"  {i}/{len(roots)}", file=sys.stderr)

    if args.list:
        want = set(args.list.split("+"))
        for r in rows:
            if want <= set(r["status"].split("+")):
                print(f"{r['work']}\t{r['pages']}")
        return

    LOGS.mkdir(exist_ok=True)
    cols = ["work", "status", "scans", "transcribed", "last_page", "to_do", "gaps", "no_ocr",
            "no_ocr_pages", "failed", "reason"]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    count, pages = Counter(), Counter()
    for r in rows:
        for c in r["status"].split("+"):
            count[c] += 1
            pages[c] += r["to_do"]
    print(f"{len(rows)} work(s) with something missing -> {OUT}")
    for c, k in count.most_common():
        print(f"  {c:10} {k:5} work(s), {pages[c]:7} page(s) to transcribe")
    print(f"  page(s) to transcribe: {sum(r['to_do'] for r in rows)}")
    unsplit = [r for r in rows if r["no_ocr"]]
    if unsplit:
        print(f"  page(s) transcribed without an OCR layer (not split into lines, no_ocr.txt): "
              f"{sum(r['no_ocr'] for r in unsplit)} in {len(unsplit)} work(s)")
    if args.detail:
        for r in rows:
            if args.status and r["status"] != args.status:
                continue
            print(f"{r['status']:12} {r['transcribed']:5}/{r['scans']:<5} {r['work']}")
            if r["to_do"]:
                print(f"             to do:   {r['gaps']}")
            if r["no_ocr"]:
                print(f"             no OCR:  {r['no_ocr_pages']}")
            if r["reason"]:
                print(f"             failed:  {r['reason']}")


if __name__ == "__main__":
    main()
