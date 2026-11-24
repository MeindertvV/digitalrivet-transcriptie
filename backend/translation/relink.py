"""Link every translation to the blocks of its book as it is now (blocks.py), without a model.

Run it after a new transcription; translate.py does the same for the book and language it is given.
Without --write it only looks: nothing is changed, and the report says what a run would do and how
many words would have to be translated (again).

A book whose translation was made on an older transcription can lose much of it: the text is the
same passage in other words or another cut, so it finds no block, and what is retired leaves the
reader. Such a book stays as it is (held()) and is named in the report, until --force.

Examples
  python translation/relink.py                             # the whole library, a look only
  python translation/relink.py W06290 W01206 --write       # these works
  python translation/relink.py --write                     # the whole library, but for the books it would cost
"""

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.paths import DATA, refuse_foreign  # noqa: E402
from translation import blocks  # noqa: E402
from common.units import plain, read_units  # noqa: E402
from common.works import ids_of  # noqa: E402

KEEP = "before_blocks"   # the translations as they were, once: <work>/before_blocks/translation_<lang>.json
HOLD, HOLD_UNITS = 0.03, 5   # a book that would retire this share and this many units: more than a cover sheet


def langs(book: Path) -> list[Path]:
    """The translations of a book. The Opera Calvini (oc) has the form of one but is made per run."""
    return [f for f in sorted(book.glob("translation_*.json")) if f.stem != "translation_oc"]


def held(count: dict) -> bool:
    """Would linking cost this book more of its translation than a cover sheet?"""
    have = sum(count[k] for k in ("same", "near", "changed", "retired"))
    return count["retired"] >= HOLD_UNITS and count["retired"] >= HOLD * have


def relink(book: Path, write: bool, force: bool = False) -> dict:
    """Counts of one book: same, near, changed, retired, missing, words: what is left to translate,
    and held: 1 if it is left as it is."""
    units = read_units(book / "book.xml")
    version = blocks.sync(book, units, False)["version"]
    out = {"work": book.name.split("_")[0], "units": 0, "same": 0, "near": 0, "changed": 0, "retired": 0, "missing": 0, "words": 0, "held": 0}
    linked = []
    for f in langs(book):
        data = json.loads(f.read_text(encoding="utf-8"))
        raw = json.dumps(data["units"], ensure_ascii=False)
        count = blocks.link(data, units, version)
        done = {t["id"] for t in data["units"] if not t.get("stale")}
        todo = [u for u in units if u.id not in done]
        for k, n in count.items():
            out[k] += n
        out["units"] += len(units)
        out["missing"] += len(todo) - count["changed"]
        out["words"] += sum(len(plain(u.la).split()) for u in todo)
        linked.append((f, data, raw))
    out["held"] = int(not force and held(out))
    if write and not out["held"]:
        blocks.sync(book, units, True)
        for f, data, raw in linked:
            if json.dumps(data["units"], ensure_ascii=False) != raw:
                keep = book / KEEP / f.name
                if not keep.exists():
                    keep.parent.mkdir(exist_ok=True)
                    keep.write_bytes(f.read_bytes())
                blocks.save(f, data)
    return out


def one(item):
    book, write, force = item
    try:
        return relink(book, write, force)
    except (OSError, ValueError, KeyError) as e:   # a file being written by another run: next time
        return {"work": book.name.split("_")[0], "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="work ids (default: every work with a translation)")
    ap.add_argument("--write", action="store_true", help="write blocks.json and the translations")
    ap.add_argument("--force", action="store_true", help="also the books that would lose part of their translation")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tsv", type=Path, help="also write the counts per work to this file")
    args = ap.parse_args()
    if args.write:
        refuse_foreign(DATA)
    want = ids_of(args.works)
    books = [d for d in sorted(DATA.iterdir()) if d.is_dir() and (d / "book.xml").is_file() and langs(d)
             and (not want or d.name.split("_")[0] in want)]
    total, rows, errors = collections.Counter(), [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(pool.map(one, [(b, args.write, args.force) for b in books], chunksize=4), 1):
            if "error" in r:
                errors.append(r)
                continue
            rows.append(r)
            total.update({a: b for a, b in r.items() if a != "work"})
            if k % 100 == 0:
                print(f"{k}/{len(books)}", flush=True)
    if args.tsv:
        keys = list(rows[0])
        args.tsv.write_text("\n".join("\t".join(str(r[c]) for c in keys) for r in [dict(zip(keys, keys)), *rows]) + "\n", encoding="utf-8")
    print(f"{len(rows)} works, {total['units']} units in the books")
    for k in ("same", "near", "changed", "retired", "missing"):
        print(f"{total[k]:9d}  {k}")
    print(f"{total['words']:9d}  words to translate (again)")
    if total["held"]:
        print(f"{total['held']:9d}  works held: linking would retire part of their translation (--force)")
        print("           " + " ".join(r["work"] for r in rows if r["held"]))
    for e in errors:
        print(f"not read: {e['work']}: {e['error']}")


if __name__ == "__main__":
    main()
