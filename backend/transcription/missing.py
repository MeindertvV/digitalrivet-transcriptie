"""Which pages has transcribe.py skipped? Read every pNNNN.xml that is a missing-page mark, count
them by reason, and say whether a mark is stale (the page can be split again now).

  python transcription/missing.py                 # summary
  python transcription/missing.py --list W05990_... # the marked pages of one work
  python transcription/missing.py --clear-stale   # remove a mark whose page can be split again
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/
from common.paths import DATA, PDF_DIR  # noqa: E402

MARK = '<page type="missing"'
REASON = re.compile(r"was not transcribed: (.*?)</warning>", re.S)


def marked(root: Path) -> dict[int, str]:
    d = root / "pages"
    if not d.exists():
        return {}
    out = {}
    for f in d.glob("p*.xml"):
        if not f.stem[1:].isdigit() or "para" in f.stem:
            continue
        head = f.read_text(encoding="utf-8", errors="ignore")
        if head.lstrip().startswith(MARK):
            m = REASON.search(head)
            out[int(f.stem[1:])] = " ".join((m.group(1) if m else "").split())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", help="list the marked pages of this work")
    args = ap.parse_args()
    if args.list:
        for n, why in sorted(marked(DATA / args.list).items()):
            print(f"{n}\t{why}")
        return
    total, works, why = 0, 0, Counter()
    for root in sorted(p for p in DATA.iterdir() if p.is_dir()):
        m = marked(root)
        if not m:
            continue
        works += 1
        total += len(m)
        for r in m.values():
            why[r] += 1
        if not (root / "book.json").exists():
            print(f"no book.json: {root.name}")
    print(f"{total} marked page(s) in {works} work(s)")
    for r, k in why.most_common():
        print(f"  {k:5}  {r}")


if __name__ == "__main__":
    main()
