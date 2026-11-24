"""Finish the translations that stopped when the Gonka credit was spent (REFACTOR_PLAN.md P12).

On 3 November 2026 the credit ran out during a run, and 176 works were left without a translation or with half of
one: their translation_<lang>/failed.txt names "HTTP 402" for every group that was refused. This asks those groups
again, work by work, with translate.py, and no others: every call names its groups (--groups).

  ./rivet translate_finish --dry-run          the works, their groups and the words still to do; no model, nothing written
  ./rivet translate_finish                    GLM via Gonka, two works at a time; stops when the credit is spent
  ./rivet translate_finish --via api          Haiku, only for the groups GLM was asked and could not do
  ./rivet translate_finish --only W00018 W00027 --hours 6

The list is made once and kept (logs/translate_finish_<lang>.json): translate.py writes failed.txt anew at every
run, so after the first call a work's failed.txt no longer says what was refused. Per work the file has the groups
(by the id of their first unit, as failed.txt names them), whether GLM and Haiku have been asked, and what was left
after each. Run it again to continue: a work that has been asked is passed by, and translate.py skips what is
translated. A work that may not be translated (a duplicate: common/dubbel.py) is passed by and named.

A group is --words 250 on both routes, so that a group has the same number for GLM and for Haiku.
Ends with 0 (all asked), 1 (translate.py stopped: the credit is spent, no key, a usage limit; what it said is in the
output above the last line) or 75 (stopped at --hours with works left).
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import dubbel  # noqa: E402
from common import paths  # noqa: E402
from common.units import read_units  # noqa: E402

TRANSLATE = [sys.executable, str(Path(__file__).with_name("translate.py"))]
WORDS = 250             # Latin words per group, as translate.py takes on Gonka
NO_CREDIT = "HTTP 402"  # what the broker answered when the credit was spent
NOT_DONE = 75


def plan_path(lang: str) -> Path:
    return paths.LOGS / f"translate_finish_{lang}.json"


def refused(data: Path, lang: str) -> dict[str, list[str]]:
    """{work folder: [group ids]} of the groups that failed.txt says were refused for lack of credit."""
    out = {}
    for f in sorted(data.glob(f"W*/translation_{lang}/failed.txt")):
        gids = [line.split("\t")[0] for line in f.read_text(encoding="utf-8").splitlines() if NO_CREDIT in line]
        if gids:
            out[f.parent.parent.name] = gids
    return out


def load_plan(data: Path, lang: str) -> dict[str, dict]:
    """The kept list, with the works and groups added that a failed.txt names as refused now."""
    path = plan_path(lang)
    plan = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for work, gids in refused(data, lang).items():
        entry = plan.setdefault(work, {"groups": [], "glm": False, "haiku": False, "left": []})
        new = [g for g in gids if g not in entry["groups"]]
        if new:   # refused again after the list was made: GLM is asked once more
            entry["groups"] += new
            entry["glm"] = entry["haiku"] = False
    return plan


def save_plan(plan: dict, lang: str):
    path = plan_path(lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def groups_of(book_dir: Path, gids: list[str], lang: str) -> tuple[list[int], int]:
    """The numbers (from 1, as --groups counts) of the groups that hold one of these units, and the Latin words in
    them that have no translation yet."""
    from translation.translate import make_groups   # here: translate.py loads the model libraries
    groups = make_groups(read_units(book_dir / "book.xml"), WORDS)
    want = set(gids)
    hit = [k for k, g in enumerate(groups, 1) if any(u.id in want for u in g)]
    done = set()
    kept = book_dir / f"translation_{lang}.json"
    if kept.is_file():
        done = {u["id"] for u in json.loads(kept.read_text(encoding="utf-8")).get("units", []) if u.get("tr")}
    return hit, sum(u.words() for k in hit for u in groups[k - 1] if u.id not in done)


def spec(numbers: list[int]) -> str:
    """[1, 2, 3, 7, 9, 10] as "1-3,7,9-10": the way --groups reads them."""
    runs = []
    for n in sorted(set(numbers)):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def todo(plan: dict, via: str, only: list[str]) -> list[tuple[str, list[str]]]:
    """[(work folder, group ids)] still to ask on this route: for GLM what was refused, for Haiku what GLM left."""
    out = []
    for work, entry in sorted(plan.items()):
        if only and work.split("_")[0] not in only:
            continue
        if via == "gonka" and not entry["glm"]:
            out.append((work, entry["groups"]))
        elif via == "api" and entry["glm"] and not entry["haiku"] and entry["left"]:
            out.append((work, entry["left"]))
    return out


def say(text: str):
    print(f"{time.strftime('%H:%M')} {text}", flush=True)


class Finish:
    def __init__(self, data: Path, lang: str, via: str, plan: dict, until: float | None):
        self.data, self.lang, self.via, self.plan, self.until = data, lang, via, plan, until
        self.lock = threading.Lock()
        self.stopped = threading.Event()   # translate.py ended with an error: no new work is begun
        self.late = False

    def one(self, item: tuple[str, list[str]]) -> str:
        work, gids = item
        if self.stopped.is_set():
            return "not begun"
        if self.until is not None and time.time() >= self.until:
            self.late = True
            return "not begun"
        book_dir = self.data / work
        numbers, words = groups_of(book_dir, gids, self.lang)
        if not numbers:
            say(f"{work}: none of its {len(gids)} group(s) is in the book any more, passed by")
            code = 0
        else:
            say(f"{work}: {len(numbers)} group(s), {words} words still to do")
            cmd = TRANSLATE + [str(book_dir), "--lang", self.lang, "--groups", spec(numbers), "--words", str(WORDS)]
            code = subprocess.run(cmd + (["--via", "api"] if self.via == "api" else [])).returncode
        failed = book_dir / f"translation_{self.lang}" / "failed.txt"
        left = [line.split("\t")[0] for line in failed.read_text(encoding="utf-8").splitlines()] if failed.is_file() else []
        with self.lock:
            entry = self.plan[work]
            if code == 0:
                entry["glm" if self.via == "gonka" else "haiku"] = True
                entry["left"] = left
                if self.via == "gonka":
                    entry["haiku"] = False
            save_plan(self.plan, self.lang)
        if code:
            self.stopped.set()
            say(f"{work}: translate.py ended with {code}; no new work is begun")
            return "stopped"
        say(f"{work}: asked, {len(left)} group(s) left")
        return "asked"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--via", choices=["gonka", "api"], default="gonka", help="gonka: GLM (the default); api: Haiku, for what GLM left")
    ap.add_argument("--only", nargs="+", default=[], metavar="WORK", help="only these works (W00018 ...)")
    ap.add_argument("--parallel", type=int, default=2, help="works at a time (each with translate.py's own --workers)")
    ap.add_argument("--hours", type=float, help="begin no new work after this many hours")
    ap.add_argument("--dry-run", action="store_true", help="say what would be asked; no model, nothing written")
    args = ap.parse_args()
    data = paths.DATA
    if not args.dry_run:
        paths.refuse_foreign(data)
    plan = load_plan(data, args.lang)
    work = []
    for name, gids in todo(plan, args.via, args.only):
        if why := dubbel.not_to_translate(data / name, args.lang):
            say(f"{name}: passed by, {why}")
        elif not (data / name / "book.xml").is_file():
            say(f"{name}: passed by, no book.xml")
        else:
            work.append((name, gids))
    if args.dry_run:
        groups = words = 0
        for name, gids in work:
            numbers, w = groups_of(data / name, gids, args.lang)
            print(f"{name[:60]:<60} {len(numbers):6d} groups {w:9d} words")
            groups, words = groups + len(numbers), words + w
        print(f"{len(work)} work(s), {groups} group(s), {words} Latin words still to do ({args.via}); "
              f"{len(plan)} work(s) on the list")
        return 0
    save_plan(plan, args.lang)
    say(f"{len(work)} work(s) to ask ({args.via}), {args.parallel} at a time")
    run = Finish(data, args.lang, args.via, plan, time.time() + 3600 * args.hours if args.hours is not None else None)
    with ThreadPoolExecutor(args.parallel) as pool:
        ends = list(pool.map(run.one, work))
    say(f"{ends.count('asked')} asked, {ends.count('stopped')} stopped, {ends.count('not begun')} not begun")
    return 1 if run.stopped.is_set() else NOT_DONE if run.late else 0


if __name__ == "__main__":
    sys.exit(main())
