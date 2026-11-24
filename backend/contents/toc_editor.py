"""The editor of a book's contents: a second pass of Claude Haiku over the outline toc_context.py made.

toc_context.py finds the divisions and where they begin; many come out as a bare label ("Caput III.",
"Sect. 12."). A scholar wants to see what each division is about, so this pass gives a bare label its subject,
from the best source there is, and says which:
  toc    the book's printed table of contents (Haiku reads it as a list, code links each item to a division
         by label, number and parent, in the table's order: linked(); a table numbered one off against the
         headings is set right by the words they share with the text: unshift());
  head   the division's own printed title or argument just below the heading;
  lemma  the Bible text a division of a commentary or a sermon opens with;
  gist   what the opening sentences plainly state.
It also puts the titles of a series in one form, and adds the divisions the printed table lists and the outline
lacks, where their label and number stand on a page (completed()).

What it may not do (applied(), each from a wrong answer in the blind checks of 13 November 2026): cut a title
down, take a 'toc' subject from anything but the linked item, give a subject that only repeats the title, change
a level or drop an entry (it only reports those, in the kept answer).

The outline it reads is toc_context's result (data/<work>/toc_context.json, or --drafts <folder>/<work>.json);
it writes data/<work>/toc_editor.json with the same entries plus "subject" and "src" ("added" for what came
from the table), and its answers under data/<work>/toc_editor/; --out writes both elsewhere, for a trial.
Nothing reads the result yet: structure.py takes it once its form on the site is decided. Through the API,
with ANTHROPIC_API_KEY; Haiku only (common/llm.py).

Examples
  python toc_editor.py W00660 --drafts ~/.cache/toc_context/n09/run --out ~/.cache/toc_context/n09/edit
  python toc_editor.py --list works.txt --drafts ... --out ... --parallel 4   # a book that has its result is skipped
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.env import saved_env  # noqa: E402
from common.llm import MODEL_HAIKU, client, cost_of  # noqa: E402
from common.paths import PROMPTS  # noqa: E402
from common.text import element_text as plain_text, roman_any  # noqa: E402
from contents import headings as T  # noqa: E402

PROMPT = PROMPTS / "toc_editor.txt"
TABLE_PROMPT = PROMPTS / "toc_editor_table.txt"
DRAFT_NAME = "toc_context.json"
OUT_NAME = "toc_editor.json"
RAW = "toc_editor"
WORDS = 45            # words of the text after a heading shown with each entry
SHIFT_WORDS = 250     # ... and compared with the table's items to see whether they are one off
SHIFT_MIN = 4         # linked siblings in a row from which a shift says something
SHIFT_GAIN = 1.3      # how much better a shift must fit than the plain link
CHUNK = 100           # entries per request: with more, Haiku gave subjects to few of them (Bullinger: 8 of 295)
TABLE_CHUNK = 25000   # characters of the printed table per request
TABLE_MOST = 120000   # characters of the printed table at most
MAX_TOKENS = 64000    # an answer, thinking included (Haiku 5.5 allows 128K; at 32K Blondel's ran out)
EFFORT = "medium"     # Haiku 5.5's default, set so that it stays
SHORTER = 0.6         # a new title below this share of the old one is a cut, not a repair
LABEL = re.compile(r"^\s*(?:[§¶]\s*|([^\W\d_]{2,}\.?)\s+)([IVXLCDMJivxlcdmj]+|\d+)\b")
TABLE_TITLE = re.compile(r"\b(index|indice|elenchus|syllabus|table|tabula|catalogus|conspectus|series)\b.*"
                         r"\b(cap|chap|libr|quaest|disp|sect|homil|conc|serm|tract|materi)", re.I)
NOT_TABLE = re.compile(r"rerum|locorum|verborum|alphab|auctor|matieres|passages", re.I)


def plain(el) -> str:
    return " ".join(plain_text(el).split())


def num(x) -> int | None:
    """A printed number, roman (also with j and u) or arabic."""
    x = (x or "").strip().strip(".").upper().replace("J", "I").replace("U", "V")
    return int(x) if x.isdigit() else roman_any(x)


def key(title: str) -> tuple[str, int] | None:
    """The kind of label (its first three letters) and the number a heading starts with."""
    m = LABEL.match(title or "")
    if not m:
        return None
    n = num(m.group(2))
    return ((m.group(1) or "§").lower()[:3], n) if n is not None else None


def words(t: str) -> set[str]:
    return {x.lower()[:6] for x in re.findall(r"\w{5,}", t or "")}


def after(pages: dict, e: dict, most: int = WORDS) -> str:
    """The first words of the text after the heading lines of an entry. An entry found on a scan without
    its line starts after the line that carries its number."""
    order = sorted(pages)
    s = e["scan"]
    i = e["lines"][-1][1] + 1 if e.get("lines") else 0
    k = key(e["title"])
    if not e.get("lines") and s in pages and k:
        for j, t in enumerate(plain(l) for l in T.lines_of(pages[s])):
            m = re.match(r"^\W*(?:[^\W\d_]{2,}\.?\s+)?([IVXLCDMJ]+|\d+)\b", t)
            if m and num(m.group(1)) == k[1]:
                i = j + 1
                break
    out, at = [], order.index(s) if s in pages else 0
    while len(out) < most and at < len(order):
        out += " ".join(plain(l) for l in T.lines_of(pages[order[at]])[i:]).split()
        at, i = at + 1, 0
    return " ".join(out[:most])


def table_pages(toc: list[dict], pages: dict) -> dict[int, str]:
    """The pages that may be the printed table of contents: at the front and back as headings.py looks for
    them, the pages that go on with them, and the pages of an entry the outline calls a table of chapters."""
    order = sorted(pages)
    tail = max(T.TAIL, len(order) // 10)
    found = {}
    for k, s in enumerate(order):
        pg = pages[s]
        if T.HEAD_PAGES <= k < len(order) - tail and not (pg.get("type") == "contents" and order[k - 1] in found):
            continue
        ls = [plain(l) for l in T.lines_of(pg)]
        top = " ".join([pg.findtext("running_head") or ""] + ls[:T.TOP])
        if pg.get("list") != "index" and (pg.get("type") == "contents" or (T.CONTENTS.search(top) and not T.ALPHABETICAL.search(top))):
            found[s] = "\n".join(l[:120] for l in ls if T.WORD.search(l))
    for j, e in enumerate(toc):
        if TABLE_TITLE.search(e["title"]) and not NOT_TABLE.search(e["title"]):
            end = toc[j + 1]["scan"] if j + 1 < len(toc) else e["scan"] + 1
            for s in range(e["scan"], min(end + 1, e["scan"] + 12)):
                if s in pages and s not in found:
                    found[s] = "\n".join(plain(l)[:120] for l in T.lines_of(pages[s]))
    return dict(sorted(found.items()))


def linked(toc: list[dict], items: list[dict]) -> dict[int, dict]:
    """The item of the printed table for each entry that has one: the same number, the same kind of label or
    none printed, under the parent with the entry's parent's number when the table gives parents, the first
    unused one after the last link (the table's order)."""
    hint, used, last, stack = {}, set(), -1, []
    parents = any(key(it.get("parent") or "") for it in items)
    for i, e in enumerate(toc):
        while stack and stack[-1][0] >= e["level"]:
            stack.pop()
        k = key(e["title"])
        anc = next((x[1] for x in reversed(stack) if x[1]), None)
        stack.append((e["level"], k))
        if not k:
            continue
        cand = [j for j, it in enumerate(items) if j not in used and num(it.get("n")) == k[1]
                and (it.get("label") or "").lower()[:3] in (k[0], "")]
        lab = [j for j in cand if (items[j].get("label") or "").lower()[:3] == k[0]]
        cand = lab or cand
        if cand and anc:
            par = [j for j in cand if key(items[j].get("parent") or "") and key(items[j]["parent"])[1] == anc[1]]
            if par:
                cand = par
            elif parents:
                continue   # the table gives parents and none is this one's
        if cand:
            j = ([j for j in cand if j > last] or cand)[0]
            hint[i] = items[j]
            used.add(j)
            last = j
    return hint


def unshift(toc: list[dict], hint: dict[int, dict], pages: dict) -> list[str]:
    """A printed table numbered one off against the headings (its item 1 is the heading's Sect. 2): per run of
    linked siblings, the items move by one when they share clearly more words with the text of the
    neighbouring entry than with their own. Changes hint; says what it moved."""
    runs, cur = [], []
    for i in sorted(hint):
        if cur and (i != cur[-1] + 1 or toc[i]["level"] != toc[cur[-1]]["level"]):
            runs.append(cur)
            cur = []
        cur.append(i)
    runs.append(cur)
    log = []
    for run in runs:
        if len(run) < SHIFT_MIN:
            continue
        text = {i: words(after(pages, toc[i], SHIFT_WORDS)) for i in run}
        score = {sh: sum(len(words(hint[i].get("title")) & text[i + sh]) for i in run if i + sh in text) for sh in (-1, 0, 1)}
        best = max(score, key=score.get)
        if best and score[best] >= SHIFT_GAIN * max(score[0], 1):
            old = {i: hint.pop(i) for i in run}
            for i in run:
                if i - best in old:
                    hint[i] = old[i - best]
            log.append(f"items of the series from {toc[run[0]]['title'][:30]!r} moved by {best} (shared words {score})")
    return log


def completed(toc: list[dict], items: list[dict], hint: dict[int, dict], pages: dict, skip=()) -> list[dict]:
    """The divisions the printed table lists and the outline lacks: an item with a label the outline uses, a
    number and a page, placed on that printed page (or the next) between its linked neighbours where a short
    line near the top carries that label and number. Never on the pages of the table itself (skip)."""
    levels = {}
    for e in toc:
        k = key(e["title"])
        if k:
            levels.setdefault(k[0], []).append(e["level"])
    at = {id(it): toc[i]["scan"] for i, it in hint.items()}
    order = sorted(pages)
    by_page = {}
    for s in order:
        p = num(pages[s].findtext("page_number"))
        if p is not None:
            by_page.setdefault(p, []).append(s)
    added = []
    for j, it in enumerate(items):
        lab, n, p = (it.get("label") or "§").lower()[:3], num(it.get("n")), num(it.get("page"))
        if id(it) in at or lab not in levels or n is None:
            continue
        lo = max([at[id(x)] for x in items[:j] if id(x) in at] or [0])
        hi = min([at[id(x)] for x in items[j + 1:] if id(x) in at] or [10 ** 6])
        if p is None:
            continue   # without its printed page an item is not placed: searched for, it landed in another book's chapter (W00592)
        where = [s for s in by_page.get(p, []) + by_page.get(p + 1, []) if lo < s < hi]
        for s in where:
            if s in skip or pages[s].get("type") == "contents":
                continue
            lines = [plain(l) for l in T.lines_of(pages[s])]
            hit = [i for i, t in enumerate(lines) if len(t) < 90 and key(t) == (lab, n) and (i < 4 or len(t) < 40)]
            if hit:
                level = max(set(levels[lab]), key=levels[lab].count)
                title = f"{(it.get('label') or '').strip('.').capitalize()} {(it.get('n') or '').strip('.')}.".strip()
                added.append({"scan": s, "line": hit[0], "lines": [[s, hit[0]]], "level": level, "title": title,
                              "subject": it.get("title"), "src": "toc", "added": True})
                break
    return added


def applied(toc: list[dict], answer: dict[int, dict], hint: dict[int, dict]) -> list[dict]:
    """The outline with the editor's answer, within what it may do."""
    out = []
    for i, e in enumerate(toc):
        c = answer.get(i, {})
        title = c.get("title") or e["title"]
        if len(title) < SHORTER * len(e["title"]) and len(e["title"]) > 25:
            title = e["title"]   # a title is not cut down (Bochart: argument sentences reduced to their numeral)
        subject, src = c.get("subject"), c.get("src")
        if src == "toc" and i not in hint:
            subject = None       # 'toc' only from the linked item (Blondel: an item of the next section)
        if subject and words(subject) and words(subject) <= words(title):
            subject = None       # it only repeats the title
        new = {**e, "title": title}
        if subject:
            new.update(subject=subject, src=src)
        out.append(new)
    return out


def inserted(res: list[dict], added: list[dict]) -> list[dict]:
    """The added entries in the outline, each before the first entry after it; the outline keeps its order."""
    for a in sorted(added, key=lambda e: (e["scan"], e["line"])):
        k = next((i for i, e in enumerate(res) if (e["scan"], e.get("line") or 0) > (a["scan"], a["line"])), len(res))
        res.insert(k, a)
    return res


def json_lines(text: str):
    for line in text.splitlines():
        line = line.strip().strip(",")
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except ValueError:
                pass


def ask(cl, system: str, user: str, raw: Path, name: str) -> tuple[str, float, bool]:
    """One request, kept under raw/<name>_<hash>.json by what was asked: asked again, it costs nothing.
    Also whether the answer was cut off at MAX_TOKENS."""
    h = hashlib.sha256((MODEL_HAIKU + system + user).encode()).hexdigest()[:20]
    kept = raw / f"{name}_{h}.json"
    if kept.is_file():
        k = json.loads(kept.read_text(encoding="utf-8"))
        return k["text"], 0.0, k["stop"] == "max_tokens"
    for attempt in range(5):
        try:
            with cl.messages.stream(model=MODEL_HAIKU, max_tokens=MAX_TOKENS, system=system,
                                    output_config={"effort": EFFORT},
                                    messages=[{"role": "user", "content": user}]) as st:
                r = st.get_final_message()
            break
        except Exception as e:   # overloaded, a dropped connection: wait and ask again
            if attempt == 4:
                raise
            print(f"  {name}: {type(e).__name__}: {str(e)[:100]}; again", flush=True)
            time.sleep(30 * (attempt + 1))
    text = "".join(b.text for b in r.content if b.type == "text")
    cost = cost_of(MODEL_HAIKU, r.usage, False)
    kept.write_text(json.dumps({"text": text, "stop": r.stop_reason, "cost": cost, "input": r.usage.input_tokens,
                                "output": r.usage.output_tokens}, ensure_ascii=False), encoding="utf-8")
    return text, cost, r.stop_reason == "max_tokens"


def table_items(cl, tables: dict[int, str], raw: Path) -> tuple[list[dict], float]:
    """The printed table as items, read by Haiku in parts of TABLE_CHUNK characters."""
    system = TABLE_PROMPT.read_text(encoding="utf-8")
    parts, cur = [], ""
    for s, text in tables.items():
        page = f"[scan {s}]\n{text}\n"
        if cur and len(cur) + len(page) > TABLE_CHUNK:
            parts.append(cur)
            cur = ""
        cur += page
    parts.append(cur)
    items, cost = [], 0.0
    budget = TABLE_MOST
    for k, part in enumerate(p for p in parts if p.strip()):
        if budget <= 0:
            break
        budget -= len(part)
        text, c, _ = ask(cl, system, part, raw, f"table{k}")
        items += [it for it in json_lines(text) if isinstance(it, dict)]
        cost += c
    return items, cost


def run(work: str, cl, drafts: Path | None = None, out: Path | None = None, redo: bool = False) -> dict | None:
    folder = T.folder_of(work)
    target = out / f"{work}.json" if out else folder / OUT_NAME
    if target.is_file() and not redo:
        return None
    draft_file = drafts / f"{work}.json" if drafts else folder / DRAFT_NAME
    if not draft_file.is_file():
        print(f"{work}: no outline of toc_context ({draft_file})", flush=True)
        return None
    draft = json.loads(draft_file.read_text(encoding="utf-8"))
    toc = draft["toc"]
    raw = (out / "raw" / work) if out else folder / RAW
    raw.mkdir(parents=True, exist_ok=True)
    root = etree.parse(str(folder / "book.xml"), etree.XMLParser(recover=True)).getroot()
    pages = {int(p.get("n")): p for p in root.iter("page")}
    tables = table_pages(toc, pages)
    items, cost = table_items(cl, tables, raw) if tables else ([], 0.0)
    hint = linked(toc, items)
    moved = unshift(toc, hint, pages)
    rows = []
    for i, e in enumerate(toc):
        p = (pages[e["scan"]].findtext("page_number") or "").strip() if e["scan"] in pages else ""
        item = f" TOC: {hint[i]['title'][:200]}" if i in hint and hint[i].get("title") else ""
        rows.append(f"{i} L{e['level']} p.{p or '?'} {e['title'][:200]}{item} >> {after(pages, e)}")
    system = PROMPT.read_text(encoding="utf-8")
    head = (f"The book: {draft.get('book') or ''}\n\n"
            + (f"Its printed table of contents was read: {len(items)} items, {len(hint)} linked to an entry.\n\n"
               if items else "No printed table of contents was found.\n\n") + "The outline:\n")
    answer, notes = {}, []
    todo = [(k, min(k + CHUNK, len(rows))) for k in range(0, len(rows), CHUNK)]
    while todo:
        a, b = todo.pop(0)
        text, c, cut = ask(cl, system, head + "\n".join(rows[a:b]), raw, f"edit{a}-{b}")
        cost += c
        if cut and b - a > 10:   # the answer ran out: ask for each half
            todo[:0] = [(a, (a + b) // 2), ((a + b) // 2, b)]
            continue
        for o in json_lines(text):
            if isinstance(o.get("id"), int) and a <= o["id"] < b:
                answer[o["id"]] = o
                if o.get("drop") or o.get("flag") or o.get("level") not in (None, toc[o["id"]]["level"]):
                    notes.append(o)
    added = completed(toc, items, hint, pages, skip=set(tables))
    result = {"work": folder.name, "made": dt.datetime.now().isoformat(timespec="seconds"), "model": MODEL_HAIKU,
              "prompt": hashlib.sha256(PROMPT.read_bytes() + TABLE_PROMPT.read_bytes()).hexdigest()[:16],
              "draft": draft.get("made"), "cost_usd": round(cost, 5), "table_items": len(items), "linked": len(hint),
              "moved": moved, "notes": notes, "book": draft.get("book"),
              "toc": inserted(applied(toc, answer, hint), added)}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    subjects = sum(1 for e in result["toc"] if e.get("subject"))
    print(f"{work}: {len(toc)} entries, {subjects} subjects, {len(added)} added, table {len(items)} items "
          f"({len(hint)} linked{', moved' if moved else ''}), ${cost:.3f}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="work ids (W00660)")
    ap.add_argument("--list", type=Path, help="a file with work ids")
    ap.add_argument("--drafts", type=Path, help="the outlines of toc_context.py as <folder>/<work>.json (else with the book)")
    ap.add_argument("--out", type=Path, help="write the results and answers here, not with the book")
    ap.add_argument("--parallel", type=int, default=4, help="books at once")
    ap.add_argument("--force", dest="redo", action="store_true", help="also the books that have their result")
    args = ap.parse_args()
    key_ = saved_env("ANTHROPIC_API_KEY")
    if not key_:
        sys.exit("ANTHROPIC_API_KEY is not set")
    cl = client(api_key=key_, max_retries=4)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    works = args.works + (args.list.read_text().split() if args.list else [])

    def one(w):
        try:
            return run(w, cl, args.drafts, args.out, args.redo)
        except Exception as e:   # one book's trouble is no reason to stop the others
            print(f"{w}: {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(args.parallel) as pool:
        results = [r for r in pool.map(one, works) if r]
    print(f"done: {len(results)} books, ${sum(r['cost_usd'] for r in results):.2f}", flush=True)


if __name__ == "__main__":
    main()
