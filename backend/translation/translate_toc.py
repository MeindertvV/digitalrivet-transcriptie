"""Translate the reader's contents list (the headings in the sidebar) of the built editions.

The titles come from frontend/data/editions/<W-id>/meta.json ("toc"); the result is
data/<werk>/toc_<lang>.json, {"lang", "model", "titles": {Latin title: translation}}. It is keyed on the
Latin title, so a rebuilt contents list that keeps a title keeps its translation. build_reader.py puts
the translations in meta.json ("toc_tr") and the reader shows them with the translation chosen.
Only Haiku translates (CLAUDE.md); a language is done for every work that has translation_<lang>.json.

  python translation/translate_toc.py                    # every work with a translation, both languages
  python translation/translate_toc.py W00345 W01206      # these works (id or folder name)
  python translation/translate_toc.py --lang nl --dry    # count only
  python translation/translate_toc.py --via claude       # on the Claude subscription instead of the API

Opnieuw draaien = hervatten: titels die er al zijn, worden overgeslagen (--force: alles opnieuw).
"""

import argparse
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/
from common import llm, via_claude  # noqa: E402
from common.env import saved_env  # noqa: E402
from common.paths import DATA, PROMPTS, refuse_foreign  # noqa: E402
from common.works import ids_of  # noqa: E402
from common.llm import cost_of  # noqa: E402
from translation.translate import LANGUAGES, about, foreign_script  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
EDITIONS = BACKEND.parent / "frontend" / "data" / "editions"
CHUNK = 80          # titles per request
MODEL = llm.MODEL_HAIKU


def latin_titles(wid: str) -> list[str]:
    meta = json.loads((EDITIONS / wid / "meta.json").read_text(encoding="utf-8"))
    out = [" ".join(html.unescape(e[1]).split()) for e in meta.get("toc", [])]
    return list(dict.fromkeys(t for t in out if re.search(r"[^\W\d_]", t)))


def ask(client, via: str, system: str, titles: list[str]) -> tuple[dict[str, str], float]:
    text = "\n".join(f"[{k}] {t}" for k, t in enumerate(titles, 1))
    if via == "claude":
        msg = via_claude.ask(system, text, MODEL, "low")
    else:
        with client.messages.stream(
            model=MODEL, max_tokens=8000, messages=[{"role": "user", "content": text}],
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        ) as s:
            msg = s.get_final_message()
    answer = "".join(b.text for b in msg.content if b.type == "text")
    found = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"^\s*\[(\d+)\]\s*(.+)$", answer, re.M)}
    out = {t: found[k] for k, t in enumerate(titles, 1) if found.get(k) and not foreign_script(found[k])}
    return out, cost_of(msg.model, msg.usage, False) if via != "claude" else 0.0


def do_work(client, args, book_dir: Path, wid: str, lang: str) -> tuple[int, int, float]:
    path = book_dir / f"toc_{lang}.json"
    have = json.loads(path.read_text(encoding="utf-8"))["titles"] if path.is_file() and not args.force else {}
    todo = [t for t in latin_titles(wid) if t not in have]
    if not todo or args.dry:
        return len(todo), 0, 0.0
    language = LANGUAGES[lang]
    glossary = (PROMPTS / f"glossary_{lang}.txt").read_text(encoding="utf-8")
    system = (PROMPTS / "translate_toc.txt").read_text(encoding="utf-8").replace("{language}", language) \
        .replace("{about}", about(book_dir)).replace("{glossary}", glossary)
    cost = 0.0
    for k in range(0, len(todo), CHUNK):
        found, c = ask(client, args.via, system, todo[k:k + CHUNK])
        have.update(found)
        cost += c
    path.write_text(json.dumps({"lang": lang, "model": MODEL, "titles": have}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return len(todo), sum(t in have for t in todo), cost


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="*", help="work ids (W00345) or folder names; default: all with a translation")
    ap.add_argument("--lang", choices=sorted(LANGUAGES), help="default: every language the work has")
    ap.add_argument("--via", choices=["api", "claude"], default="api")
    ap.add_argument("--force", action="store_true", help="translate every title again")
    ap.add_argument("--dry", action="store_true", help="only count what is left")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if not args.dry:
        refuse_foreign(DATA)

    client = None
    if args.via == "api" and not args.dry:
        if not saved_env("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set (see transcribe.py for how to set it).")
        workspace = saved_env("ANTHROPIC_WORKSPACE_ID")
        client = llm.client(workspace, max_retries=5)

    jobs, want = [], ids_of(args.works)
    for d in sorted(DATA.iterdir()):
        wid = d.name.split("_")[0]
        if want and wid not in want:
            continue
        if not (EDITIONS / wid / "meta.json").is_file():
            continue
        for lang in ([args.lang] if args.lang else sorted(LANGUAGES)):
            if (d / f"translation_{lang}.json").is_file():
                jobs.append((d, wid, lang))
    print(f"{len(jobs)} contents list(s) to check")

    total, cost = 0, 0.0
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(do_work, client, args, d, wid, lang): (wid, lang) for d, wid, lang in jobs}
            for f in as_completed(futs):
                wid, lang = futs[f]
                try:
                    n, ok, c = f.result()
                except via_claude.UsageLimit as e:
                    sys.exit(f"{e}")
                except anthropic.AuthenticationError:
                    sys.exit("Authentication failed: set ANTHROPIC_API_KEY to a valid key.")
                except Exception as e:  # one work failing should not stop the rest; rerun resumes
                    print(f"{wid} {lang}: FAILED {type(e).__name__}: {e}")
                    continue
                total += n
                cost += c
                if n:
                    print(f"{wid} {lang}: {ok}/{n} titles{' (dry)' if args.dry else ''}  [${c:.4f}]")
    finally:
        print(f"{total} title(s) {'left' if args.dry else 'handled'}, ${cost:.4f}")


if __name__ == "__main__":
    main()
