"""The book's table of contents from its printed index pages, as Claude Haiku read it once: reading the
data/<work>/index_toc.json that was made, and locating its entries in the book as it is now (toc()).

REFACTOR_PLAN.md step 38: the reading of the scans itself (index_toc.py with its prompt, Haiku first and Sonnet
for the pages Haiku could not read) is in archive/index_toc/; one book has an index_toc.json, and structure.py
still takes it before every other source. What stands here is line for line what that script had for it.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contents import printed_contents as R  # noqa: E402
from common.paths import DATA  # noqa: E402

OUT_NAME = "index_toc.json"
MIN_ENTRIES = 3     # fewer located entries: no contents list


def scans_of(root) -> list[tuple[int, int | None]]:
    """[(scan index, printed page number)] as register.locate wants them."""
    return [(int(p.get("n")), R.printed_int(p.findtext("page_number"))) for p in root.iter("page")]


def numbered(entries: list[dict]) -> list[list]:
    """[[level, title, page number or None]]: a heading without a number gets that of the first
    entry under it; roman numbers and unread ones stay None."""
    out = []
    for i, e in enumerate(entries):
        ref = int(e["page"]) if re.fullmatch(r"\d{1,4}", e["page"]) else None
        if ref is None and not e["page"]:
            ref = next((int(f["page"]) for f in entries[i + 1:]
                        if f["level"] > e["level"] and re.fullmatch(r"\d{1,4}", f["page"])), None)
        out.append([e["level"], R.cut(e["title"]), ref])
    return out


def in_order(found: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The longest run of [(entry index, scan)] whose scans do not go down: an entry found out of
    order (a misprinted number that points to a page of another part) goes."""
    best: list[list[tuple[int, int]]] = []
    for item in found:
        chain = max((b for b in best if b[-1][1] <= item[1]), key=len, default=[])
        best.append(chain + [item])
    return max(best, key=len, default=[])


def fill(es: list[list], at: dict[int, int]) -> dict[int, int]:
    """Scans for the entries locate did not find, where the book's own page numbers are missing (pages
    not transcribed): between the entries found before and after, by its number. An entry without a
    number right under a group heading gets the heading's scan, another one that of the entry after it."""
    out = dict(at)
    for i, (level, _, ref) in enumerate(es):
        if i in at:
            continue
        if ref is None:
            if i and i - 1 in out and es[i - 1][0] < level:
                out[i] = out[i - 1]
            elif i + 1 in at and (i == 0 or out.get(i - 1, 0) <= at[i + 1]):
                out[i] = at[i + 1]
            continue
        before = next(((es[j][2], at[j]) for j in range(i - 1, -1, -1) if j in at and es[j][2] is not None), None)
        after = next(((es[j][2], at[j]) for j in range(i + 1, len(es)) if j in at and es[j][2] is not None), None)
        if before and after and before[0] < ref < after[0] and before[1] <= after[1]:
            out[i] = before[1] + round((ref - before[0]) * (after[1] - before[1]) / (after[0] - before[0]))
    return out


def located(es: list[list], scans) -> list[tuple[int, int, int, str]]:
    """[(entry index, level, scan, title)] of the entries register.locate finds (or fill estimates),
    in list order."""
    tagged = [[level, f"{i}\x00{title}", ref] for i, (level, title, ref) in enumerate(es)]
    found = sorted((int(t.split("\x00")[0]), scan) for _, scan, t in R.locate(tagged, scans))
    at = fill(es, dict(in_order(found)))
    return [(i, es[i][0], at[i], es[i][1]) for i in sorted(at)]


def check(lst: dict, scans) -> dict:
    """Locate a contents list's entries and say whether it holds."""
    es = numbered(lst["entries"])
    refs = [e[2] for e in es if e[2] is not None]
    found = located(es, scans)
    found_n = len(R.locate(es, scans))   # by the book's own numbers, not estimated
    last = max((p for _, p, _ in R.smoothed(scans) if p is not None), default=0)
    reachable = sum(r <= last for r in refs)
    up = sum(b >= a for a, b in zip(refs, refs[1:])) / max(1, len(refs) - 1)
    share = found_n / reachable if reachable else 0.0
    return {"found": found, "monotone": round(up, 2), "found_share": round(share, 2),
            "holds": len(found) >= MIN_ENTRIES and up >= R.MONOTONE and share >= R.FOUND}


def toc_entries(lists: list[dict], scans) -> list[tuple[int, int, str]]:
    """[(level, scan, title)] of every contents list that holds, levels from 1."""
    out = []
    for lst in lists:
        if lst["kind"] == "contents" and lst.get("holds"):
            out += [(level, scan, title) for _, level, scan, title in check(lst, scans)["found"]]
    if not out:
        return []
    top = min(level for level, _, _ in out)
    return [(level - top + 1, scan, title) for level, scan, title in out]


def toc(root) -> list[tuple[int, int, str]]:
    """[(level, scan index, title)] from data/<work>/index_toc.json, located in this book.xml, or []."""
    source = root.get("source")
    if not source:
        return []
    path = DATA / Path(source).stem / OUT_NAME
    if not path.is_file():
        return []
    return toc_entries(json.loads(path.read_text(encoding="utf-8"))["lists"], scans_of(root))
