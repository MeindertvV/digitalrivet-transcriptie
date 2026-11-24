"""Blocks: a paragraph or heading of a book with an id of its own, which a new transcription keeps.

A translation hangs on a block, not on a place in the book. The place of a paragraph (scan and line,
translate.py's Unit.id) moves when a page is transcribed again; the block stays the same paragraph as
long as its text can be recognised. So data/<work>/blocks.json holds, in reading order, every block's
id with the hash and a short signature of its text, and sync() carries the ids over to the units of the
book as it is now:
- a block with the same text keeps its id, wherever it went (exact);
- so does a block that resembles it, among the blocks that lie between two exact ones (fuzzy);
- what is left gets a new id; an id that is gone is never given out again.

link() does the same for a translation: each of its units finds its block, and then it is
- same     the Latin is the same, or NEAR in letters (a reading corrected here and there): the translation holds;
- changed  the block is there but its Latin is not the same: the translation stays, marked "stale", until
           translate.py has done it again;
- retired  no block any more: it goes to "retired" in the file, out of the reader but not lost.
Each file also says which version of the blocks it was linked to ("blocks").
"""

import difflib
import json
import os
from pathlib import Path

OUT_NAME = "blocks.json"
NEAR = 0.95        # letters in common for a translation to stay as it is when the Latin changed
ALIKE = 0.8        # of the signature, for a block to be taken for the same one
LENGTH = 0.5       # at least this much of the longer one's length, for that
WINDOW = 12        # blocks looked ahead for the same one
SHORT = 12         # words: a text shorter than this is compared by its letters, not by its words
SIG_HEAD, SIG_TAIL = 200, 56   # words kept of a long text's signature


from common.text import block_key as sha, block_letters as letters, block_words as words  # noqa: E402  (step 57)


def sig(text: str) -> str:
    """What is kept of a text to know it again: a short one's letters, else the first letter of each
    word, which survives most misreadings (f for long s folded). A text without letters: nothing."""
    w = words(text)
    if len(w) < SHORT:
        return "".join(w)
    if len(w) > SIG_HEAD + SIG_TAIL:
        w = w[:SIG_HEAD] + w[-SIG_TAIL:]
    return "".join(x[0].replace("f", "s") for x in w)


def alike(a: str, b: str) -> bool:
    if not a or not b or min(len(a), len(b)) < LENGTH * max(len(a), len(b)):
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= ALIKE


def align(old: list[tuple], new: list[tuple]) -> list[tuple[int, int, bool]]:
    """Pairs of blocks in the same order: [(index in old, index in new, exact)]. old and new: [(sha, sig)]."""
    sm = difflib.SequenceMatcher(None, [x[0] for x in old], [x[0] for x in new], autojunk=False)
    pairs, i, j = [], 0, 0
    for a, b, n in sm.get_matching_blocks():
        pairs += fuzzy(old, new, i, a, j, b)
        pairs += [(a + k, b + k, True) for k in range(n)]
        i, j = a + n, b + n
    return pairs


def fuzzy(old, new, i0, i1, j0, j1) -> list[tuple[int, int, bool]]:
    """The blocks that resemble each other between two exact ones, in order."""
    out, j = [], j0
    for i in range(i0, i1):
        hit = next((k for k in range(j, min(j + WINDOW, j1)) if alike(old[i][1], new[k][1])), None)
        if hit is not None:
            out.append((i, hit, False))
            j = hit + 1
    return out


def load(book_dir: Path) -> dict:
    path = book_dir / OUT_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 0, "next": 1, "blocks": []}


def sync(book_dir: Path, units: list, write: bool = True) -> dict:
    """Give every unit (.la) its block: sets u.block and u.sha. Returns blocks.json as it is now; it
    is written when something changed (not with write=False, for a look only)."""
    data = load(book_dir)
    old = [(s, g) for _, s, g in data["blocks"]]
    new = [(sha(u.la), sig(u.la)) for u in units]
    ids = {j: data["blocks"][i][0] for i, j, _ in align(old, new)}
    next_id = data["next"]
    rows = []
    for j, u in enumerate(units):
        if j not in ids:
            ids[j] = f"b{next_id}"
            next_id += 1
        u.block, u.sha = ids[j], new[j][0]
        rows.append([ids[j], new[j][0], new[j][1]])
    if rows != data["blocks"]:
        data = {"version": data["version"] + 1, "next": next_id, "blocks": rows}
        if write:
            save(book_dir / OUT_NAME, data)
    return data


def save(path: Path, data: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def near(a: str, b: str) -> bool:
    """Is the Latin as good as the same? Most of a long text is enough to tell."""
    a, b = letters(a)[:4000], letters(b)[:4000]
    return bool(a) and difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= NEAR


def link(data: dict, units: list, version: int) -> dict:
    """Bring a translation (translation_<lang>.json as read, in place) to the units of the book
    as it is now, which sync() has given their blocks. A unit that is linked keeps its id's place in
    the book up to date; what it counts is returned: {"same", "near", "changed", "retired"}."""
    by_block = {u.block: u for u in units}
    tr = data.get("units", [])
    # a file from before the blocks: its units say what they translated, so they are found by that
    bare = [t for t in tr if "block" not in t]
    if bare:
        for i, j, _ in align([(sha(t["la"]), sig(t["la"])) for t in bare], [(u.sha, sig(u.la)) for u in units]):
            bare[i]["block"] = units[j].block
    taken, out, retired = set(), [], data.get("retired", [])
    count = {"same": 0, "near": 0, "changed": 0, "retired": 0}
    for t in tr:
        u = by_block.get(t.get("block"))
        if u is None or u.block in taken:
            t.pop("block", None)
            retired.append(t)
            count["retired"] += 1
            continue
        taken.add(u.block)
        fixed = {k: v for k, v in t.items() if k != "stale" and k != "la_sha"}
        fixed.update(id=u.id, start=u.start, end=u.end, lines=u.lines, block=u.block, la_sha=u.sha)
        if t["la"] == u.la or sha(t["la"]) == u.sha:
            kind = "same"
            fixed["la"] = u.la
        elif near(t["la"], u.la):
            kind = "near"
            fixed["la"] = u.la
        else:
            kind = "changed"
            fixed["stale"] = True
        count[kind] += 1
        out.append(fixed)
    place = {u.block: k for k, u in enumerate(units)}
    data["units"] = sorted(out, key=lambda t: place[t["block"]])
    if retired:
        data["retired"] = retired
    data["blocks"] = version
    return count
