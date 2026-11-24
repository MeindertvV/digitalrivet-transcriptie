"""Translate a transcribed book (book.xml written by transcribe.py) paragraph by paragraph.

The text is cut into units: a paragraph (from a line with para="true" to the next one)
or a heading. Each unit keeps where it stands in the original, so the translation is
aligned with the Latin by nature: a frontend can set every paragraph beside its Latin.
Several paragraphs (about --words words of one chapter) go into one request, for
context; the model answers per unit. Margin notes (nearly all Scripture references)
are not translated.

Output: <book folder>/translation_<lang>.json
  {"book", "lang", "language", "model",
   "running_heads": {Latin running head: translation},
   "units": [{"id": "33.12", "start": [33, 12], "end": [34, 9],
              "lines": [[33, 12, 0], [33, 13, 41], ..., [34, 9, 1210]], "heading": 0,
              "la": "Latin, italics as <i>...</i>", "tr": "translation, same italics", "req": "33.12"}, ...]}
  id/start/end: [scan page, line n] in book.xml / book.json; lines: every printed line of the
  unit with the number of letters of "la" before it (letters as sitebuild/edition/build_reader.py counts
  them: no digits, punctuation, spaces or tags), so each line's piece of the Latin is known;
  heading: 0 for a paragraph, else the heading level (1 = chapter); req: the request that
  translated it, raw/<req>.json. Units are in reading order; only translated ones.
Work files in <book folder>/translation_<lang>/: raw/ (model, tokens and cost per request),
bad/ (unusable answers), failed.txt, batches.json while a batch runs.

An answer is checked per unit: the good units are kept and only the others are asked again
(with the units next to a missing, much shorter or much longer one, whose text may have moved).
A unit with Chinese, Japanese, Korean, Indic or any other non-Latin script is rejected as well;
Greek and Hebrew are allowed, for the quotations the prompt keeps in square brackets.

The prompt describes the book from its folder name (W06290_Author_Title_1634), or from
<book folder>/about.txt if that exists.

Examples
  python translation/translate.py data/W06290_Rivet_Commentarii_in_librum_secundum_Mosis_1634 --list
  python translation/translate.py data/W06290_Rivet_Commentarii_in_librum_secundum_Mosis_1634 --lang nl --groups 5-7   # pilot
  python translation/translate.py data/W06290_Rivet_Commentarii_in_librum_secundum_Mosis_1634 --lang nl --via api --batch   # Haiku, whole book, 50% off
  python translation/translate.py data/W06290_Rivet_Commentarii_in_librum_secundum_Mosis_1634 --lang en            # GLM-5.3-Flash on Gonka (the default)

  python translation/translate.py data/W06290_Rivet_Commentarii_in_librum_secundum_Mosis_1634 --lang nl --relink       # lines only, no API

Run it again to continue: units already translated (with the same Latin) are skipped.
Route: Gonka (GLM) by default; --via api for Haiku on the Anthropic API.
Effort with Haiku: medium stays the default; in a pilot (Rivet, 6 groups) low saved 29% of the output tokens
but kept only half of the italics.
Needs the Gonka key (see common/via_gonka.py); --via api needs ANTHROPIC_API_KEY in the environment
instead (never put a key in this file).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import dubbel, llm, via_claude, via_gonka  # noqa: E402
from common.env import saved_env  # noqa: E402
from common.notice import notice_pages  # noqa: E402
from common.paths import PROMPTS, refuse_foreign  # noqa: E402
from common.bookfile import line_markup  # noqa: E402
from common.pagespec import parse_pages  # noqa: E402
# these stood here; older callers still take them from this file
from common.units import END_MARKERS, TAG, Unit, balance, letters, plain, read_units  # noqa: E402,F401

from common.languages import TRANSLATED as LANGUAGES  # noqa: E402  (step 69)
from translation.validate import (ALLOWED_RANGES, FOREIGN_SCRIPTS, LEAK, META, META_IN_UNIT, MODERN, SHIFTED,  # noqa: E402,F401
                                  UNIT_LINE, char_name, copied, foreign_script, in_ranges, modern, parse_answer,
                                  partly_copied, tidy)  # (step 78)


def make_groups(units: list[Unit], target: int) -> list[list[Unit]]:
    """Consecutive units of one chapter, about `target` words per request. A group closes once
    big enough (never right after a heading), early before a heading, or before a paragraph
    that would make it far too long; a long paragraph is a group of its own."""
    groups, cur, size = [], [], 0
    for u in units:
        if cur and ((u.heading == 1 and not END_MARKERS.match(plain(u.la)))
                    or (size >= target and not cur[-1].heading)
                    or (u.heading and size >= target * 0.6)
                    or (size + u.words() > target * 1.5 and not cur[-1].heading)):
            groups.append(cur)
            cur, size = [], 0
        cur.append(u)
        size += u.words()
    if cur:
        groups.append(cur)
    merged = []
    for g in groups:  # a few words left over (e.g. a closing "DIXI."): add them to the group before
        if merged and sum(u.words() for u in g) < 50 and not (g[0].heading == 1 and not END_MARKERS.match(plain(g[0].la))):
            merged[-1] += g
        else:
            merged.append(g)
    return merged


def chapter_titles(groups: list[list[Unit]], book_dir: Path) -> list[str]:
    """The chapter each group belongs to: from the book's contents (structure.json) where there is
    one, else from the headings of level 1."""
    path = book_dir / "structure.json"
    shape = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if shape.get("entries"):
        top = min(e[2] for e in shape["entries"])
        chapters = [(e[0], e[3]) for e in shape["entries"] if e[2] == top]
        return [next((t for scan, t in reversed(chapters) if scan <= g[0].start[0]), "Front matter") for g in groups]
    titles, title = [], "Front matter"
    for g in groups:
        for u in g:
            if u.heading == 1 and not END_MARKERS.match(plain(u.la)):
                title = plain(u.la)
                break
        titles.append(title)
    return titles


def about(book_dir: Path) -> str:
    """The book in a few words: <book folder>/about.txt, else author, title and year from
    the folder name (W06290_Rivet_Commentarii_in_librum_secundum_Mosis_1634)."""
    f = book_dir / "about.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    m = re.fullmatch(r"W\d+_([^_]+)_(.+)_(\d{4})", book_dir.name)
    return f"{m.group(1)}, {m.group(2).replace('_', ' ')} ({m.group(3)})" if m else book_dir.name.replace("_", " ")


from translation.store import Store  # noqa: E402  (step 79)
from translation.run import (SENTENCE_END, Run, build_params, complete, custom_id, prompt_sha, run_batch,  # noqa: E402,F401
                             run_sync, running_heads, split_up, tail_words, too_long_for_gonka, translate_long,
                             translate_running_heads, user_message)  # (step 80)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book_dir", type=Path, help="book folder written by transcribe.py (contains book.xml)")
    ap.add_argument("--lang", default="nl", choices=sorted(LANGUAGES))
    ap.add_argument("--groups", help="e.g. 1-5,12 (default: all); see --list")
    ap.add_argument("--words", type=int, help="target Latin words per request (default 250 on Gonka, 1000 with Haiku)")
    ap.add_argument("--model", help="default zai-org/GLM-5.3-Flash on Gonka, claude-haiku-5-5 with --via api")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--prompt", type=Path, default=PROMPTS / "translate.txt")
    ap.add_argument("--glossary", type=Path, help="default: prompts/glossary_<lang>.txt")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--retries", type=int, default=2, help="extra attempts for an unusable answer")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests in sync mode")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (50%% cheaper, slower)")
    ap.add_argument("--via", choices=["api", "claude", "gonka"], default="gonka",
                    help="api: Haiku on the Anthropic API; claude: run on a Claude subscription through the Claude Code CLI (see via_claude.py); "
                         "gonka (default): GLM on the Gonka network (see common/via_gonka.py)")
    ap.add_argument("--poll", type=int, default=60, help="seconds between batch status checks")
    ap.add_argument("--force", action="store_true", help="redo units that are already translated")
    ap.add_argument("--list", action="store_true", help="show the request groups, no API calls")
    ap.add_argument("--relink", action="store_true",
                    help="only give the translated units their printed lines from book.xml, no API calls")
    ap.add_argument("--ook-dubbel", action="store_true",
                    help="translate even a work that is a duplicate of another work's reading edition (common/dubbel.py)")
    args = ap.parse_args()
    if not args.list:
        refuse_foreign(args.book_dir)
    if not (args.list or args.relink or args.ook_dubbel) and (why := dubbel.not_to_translate(args.book_dir, args.lang)):
        sys.exit(f"niet vertalen, dubbel: {why}. Alleen de leeseditie wordt vertaald; anders --ook-dubbel.")
    gonka = args.via == "gonka"
    args.words = args.words or (250 if gonka else 1000)
    args.model = args.model or ("zai-org/GLM-5.3-Flash" if gonka else llm.MODEL_HAIKU)
    if not gonka and not llm.is_haiku(args.model):  # translations are always Haiku: Sonnet/Opus cost far too much
        sys.exit(f"translate.py: only Haiku translates (or GLM via Gonka), not {args.model}")
    args.effort = args.effort or ("low" if gonka else "medium")

    language = LANGUAGES[args.lang]
    book_xml = args.book_dir / "book.xml"
    units = read_units(book_xml)
    groups = make_groups(units, args.words)
    titles = chapter_titles(groups, args.book_dir)

    if args.list:
        sys.stdout.reconfigure(encoding="utf-8")
        for k, (g, t) in enumerate(zip(groups, titles), 1):
            print(f"{k:4d}  {g[0].id:>8}  {sum(u.words() for u in g):5d} words  {len(g):3d} units  "
                  f"pages {g[0].start[0]}-{g[-1].end[0]}  {t[:40]}")
        paras = sum(1 for u in units if not u.heading)
        print(f"{len(groups)} groups, {len(units)} units ({paras} paragraphs, {len(units) - paras} headings)")
        return

    store = Store(args.book_dir, args.lang, units)
    if args.relink:
        print(f"{store.relinked} unit(s) relinked, {sum(store.done(u) for u in units)} of {len(units)} "
              f"translated -> {store.path}")
        return
    run = Run(args.book_dir / f"translation_{args.lang}", store, groups, titles, args.force)
    if args.via == "claude":
        if args.batch:
            sys.exit("--batch needs the API; leave it out with --via claude.")
        args.workers = min(args.workers, 2)  # stay gentle with the subscription's limits
        client = None
    elif gonka:
        if args.batch:
            sys.exit("--batch needs the Anthropic API: add --via api.")
        via_gonka.key()  # stop now if there is no key
        client = None
    else:
        key = saved_env("ANTHROPIC_API_KEY")
        workspace = saved_env("ANTHROPIC_WORKSPACE_ID")
        if not key:
            sys.exit("ANTHROPIC_API_KEY is not set (see transcribe.py for how to set it).")
        client = llm.client(workspace, max_retries=5)
    fill = lambda s: s.replace("{language}", language).replace("{about}", about(args.book_dir))
    glossary = (args.glossary or PROMPTS / f"glossary_{args.lang}.txt").read_text(encoding="utf-8")
    system = fill(args.prompt.read_text(encoding="utf-8")).replace("{glossary}", glossary)
    run.prompt_sha = prompt_sha(system)

    todo = [n - 1 for n in parse_pages(args.groups, len(groups))]
    if not args.force:
        todo = [i for i in todo if not all(store.done(u) for u in groups[i])]
    try:
        if not todo:
            print("Nothing to do: all selected units are already translated.")
        else:
            print(f"Translating {len(todo)} group(s) into {language} with {args.model}, "
                  f"effort {args.effort}{', batch' if args.batch else ''}")
            (run_batch if args.batch else run_sync)(client, args, system, language, run, todo)
        heads = [t for t in running_heads(book_xml) if t not in store.running_heads]
        if heads:
            found, model = translate_running_heads(
                client, args, fill((PROMPTS / "running_heads.txt").read_text(encoding="utf-8")), heads)
            with store.lock:
                store.running_heads.update(found)
                store.models.add(model)
                store.write()
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: set ANTHROPIC_API_KEY to a valid key.")
    except via_claude.UsageLimit as e:
        sys.exit(f"{e}\nThe subscription's usage limit is reached: run again later to continue.")
    except via_gonka.NoCredit as e:
        sys.exit(f"{e}\nThe Gonka credit is spent: top it up and run again to continue.")
    finally:
        failed_txt = run.work / "failed.txt"
        if run.failed:
            failed_txt.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(run.failed.items())), encoding="utf-8")
            print(f"{len(run.failed)} group(s) failed, listed in {failed_txt}")
        else:
            failed_txt.unlink(missing_ok=True)
        print(f"Spent this run: ${run.spent:.3f}")
    print(f"{sum(store.done(u) for u in units)} of {len(units)} units translated -> {store.path}")


if __name__ == "__main__":
    main()
