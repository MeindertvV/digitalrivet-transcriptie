"""The Bible books of each commentary volume, from the volume itself (backend/docs/COMMENTAREN_DOEL.md, step 1.3).

genre_haiku reads the first ten pages, so for a work on many books it sees the first (Calvin on the five books of
Moses: Genesis), and a further volume (W01529-04) shows the answer of volume 1. The running heads and book-title
headings of the whole volume (commentary.evidence and treated, without a model) see every book, but also a book
that is no subject: an author named Johannes in the running head, "IOANN. II" read as 2 John. So:
  - a further volume whose running heads name its books on SHARE_HEADS of its pages: those books (Cocceius vol. 4:
    the New Testament); below SHARE_NONE no book: it is no commentary (Zwingli vol. 2, his treatises); else volume 1's;
  - a title that says the work treats many books (common/bible.many_books): Haiku's books and the running heads';
  - a volume where Haiku named no book: the running heads';
  - otherwise Haiku's books; where the running heads differ, the volume is on the list to look over.
Writes data/<work>/bible_books.json per transcribed volume that Haiku (or its volume 1) took for a commentary:
{"books": [...], "source": "haiku" | "heads" | "haiku+heads" | "none", "haiku": [...], "heads": [...], "spans": {book:
[first scan, last scan]}, "version": VERSION}; build_frontpage.py takes "books" over Haiku's. No model, a few
seconds per volume.

Examples
  python bible_books.py --all                   # every commentary volume; prints where the two differ
  python bible_books.py W01184 W01529-04 --show
  python bible_books.py --all --dry-run         # writes nothing
"""

import argparse
import json
import re
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.bible import BOOKS, CHAPTERS, many_books  # noqa: E402
from common.paths import DATA, FRONTEND, bible_books_of, genre_answers, refuse_foreign  # noqa: E402
from common.text import bible_fold  # noqa: E402
from contents import commentary as C  # noqa: E402

VERSION = 1
SHARE_HEADS = 0.25   # of a further volume's pages whose running head names its books, from which those are its books
SHARE_NONE = 0.10    # below which its running heads name titles, not Bible books: the volume is no commentary


def canon(books) -> list[str]:
    return [b for b in BOOKS if b in set(books)]


def decide(haiku: list[str], heads: list[str], ev: dict, many: bool, inherited: bool, pages: int) -> tuple[list[str], str]:
    """(the books, where they came from) of one volume of pages scans; see the module's doc. "none": a further
    volume whose running heads name its books on fewer than SHARE_NONE of its pages holds no commentary
    (Zwingli's Opera, vol. 2: the treatises, with ECCLESIASTES. and ACTA as titles)."""
    haiku = [b for b in haiku if b in CHAPTERS]
    if heads and inherited:
        share = sum(ev[b]["rh"] for b in heads if b in ev) / max(pages, 1)
        if share >= SHARE_HEADS:
            return canon(heads), "heads"
        if share < SHARE_NONE:
            return [], "none"
        return canon(haiku), "haiku"
    if heads and not haiku:
        return canon(heads), "heads"
    if heads and many and set(heads) - set(haiku):
        return canon(set(haiku) | set(heads)), "haiku+heads"
    return canon(haiku), "haiku"


def spans(books: list[str], ev: dict) -> dict:
    """{book: [first scan, last scan]} as the volume names it, for the books it has evidence of."""
    return {b: [ev[b]["first"], ev[b]["last"]] for b in books if b in ev}


def volumes(only: set[str]):
    """(id, folder, Haiku's answer, inherited) of every transcribed volume that Haiku, or its volume 1, took for a
    commentary on the work the title names."""
    answers = json.loads(genre_answers().read_text(encoding="utf-8"))
    for d in sorted(DATA.glob("W*_*")):
        m = re.match(r"(W\d{5})(-\d+)?_", d.name)
        if not m or not (d / "book.xml").is_file():
            continue
        wid = m.group(1) + (m.group(2) or "")
        if only and wid not in only:
            continue
        a, inherited = answers.get(wid), False
        if a is None:
            a, inherited = answers.get(m.group(1)), True
        if a and a.get("bijbel") == "commentaar" and a.get("match") == "ja":
            yield wid, d, a, inherited


def one(wid: str, folder: Path, a: dict, inherited: bool, titles: dict) -> dict:
    root = etree.parse(str(folder / "book.xml"), etree.XMLParser(recover=True)).getroot()
    C.AUTHOR[0] = bible_fold(folder.name.split("_")[1])[:5] if folder.name.count("_") >= 2 else ""
    ev = C.evidence(root)
    heads = C.treated(root, [], ev)
    many = bool(many_books(titles.get(wid.split("-")[0], "")))
    books, source = decide(a["bb"], heads, ev, many, inherited, len(root.findall(".//page")))
    return {"books": books, "source": source, "haiku": a["bb"], "heads": heads, "spans": spans(books, ev),
            "version": VERSION}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="works or volumes (W01184, W01529-04)")
    ap.add_argument("--all", action="store_true", help="every commentary volume")
    ap.add_argument("--dry-run", action="store_true", help="write nothing")
    ap.add_argument("--show", action="store_true", help="print every volume, not only where the two differ")
    args = ap.parse_args()
    if not args.works and not args.all:
        ap.error("name works, or --all")
    if not args.dry_run:
        refuse_foreign(DATA)   # the files lie with the books: written from the live folder, not the dev copy
    site = json.loads((FRONTEND / "data" / "works.json").read_text(encoding="utf-8"))
    titles = {r[0]: r[site["fields"].index("t")] or "" for r in site["works"]}
    count, differ = {}, 0
    for wid, folder, a, inherited in volumes(set(args.works)):
        r = one(wid, folder, a, inherited, titles)
        count[r["source"]] = count.get(r["source"], 0) + 1
        if not args.dry_run:
            bible_books_of(folder).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        more = set(r["heads"]) - set(r["books"])   # books the running heads name and the result has not
        differ += bool(more)
        if args.show or r["source"] != "haiku" or more:
            mark = "" if r["source"] != "haiku" else f"  (kijken: {', '.join(canon(more))})"
            print(f"{wid:10} {r['source']:12} {', '.join(r['books'])[:90]}{mark}")
            if r["source"] == "haiku" and r["heads"]:
                print(f"{'':23} kopregels: {', '.join(r['heads'])[:90]}")
    print(f"bible_books: {sum(count.values())} volumes, {count}; running heads name more in {differ}"
          f"{' (dry run: nothing written)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
