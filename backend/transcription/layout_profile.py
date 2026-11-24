"""Choose a layout standard for a work from a sample of its pages, before it is transcribed.

Haiku looks at a few random pages of the work (10, and fewer than 10% of the pages: a work of 60
pages gets 5), picks one of the standards in prompts/layout/ and adds what holds for this book
only (its languages, the kind of notes, recurring elements). The answer is saved as
data/<work>/layout_profile.json; transcribe.py --profile then reads the work with the base prompt
plus the standard, the extra modules and that note.

The standards (prompts/layout/<name>.txt, first line "# description"):
  gewoon, twee_kolommen, drie_kolommen, parallel, kanttekeningen, regels, tabel, beeld,
  grieks_hebreeuws. Any but gewoon also gets _common.txt: notes anchored on the sentence they
  stand beside (at="first words"), pictures as <figure>, Greek and Hebrew marked when illegible.
Two printed pages on one scan need no standard: parts.py cuts those.

  python transcription/layout_profile.py W00339_Aretius_... [W...]          # choose and save
  python transcription/layout_profile.py W00339_... --show                  # the prompt it gives
"""

import argparse
import base64
import json
import random
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import llm, via_claude  # noqa: E402
from common.env import saved_env  # noqa: E402
from common.paths import DATA, PDF_DIR, PROMPTS, ocr_layer, refuse_foreign  # noqa: E402
from common.works import layer_if_there  # noqa: E402

LAYOUT = PROMPTS / "layout"
BASE = PROMPTS / "early_modern_latin_paragraphs.txt"
MODEL = llm.MODEL_HAIKU         # never another model: see CLAUDE.md
SAMPLE = 10                     # at most this many pages, and fewer than 10% of the work
MAX_PX = 1568                   # long edge of each sample image: enough to see the layout
MODULES = ("kanttekeningen", "grieks_hebreeuws", "beeld", "tabel", "regels")   # may come on top of a standard
PRICE = llm.PRICES[MODEL]       # $ per million tokens, Haiku


def client_():
    saved_env("ANTHROPIC_API_KEY")
    ws = saved_env("ANTHROPIC_WORKSPACE_ID")
    return llm.client(ws, max_retries=5)


def standards() -> dict[str, str]:
    """{name: description} of prompts/layout/*.txt."""
    out = {}
    for f in sorted(LAYOUT.glob("*.txt")):
        if not f.name.startswith("_"):
            out[f.stem] = f.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
    return out


def body(name: str) -> str:
    """A standard's text without its description line."""
    lines = (LAYOUT / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("#")).strip()


def sample_pages(work: str, k: int | None = None, seed: int | None = None) -> list[int]:
    """1-based page numbers: random pages with text on them, at most SAMPLE and under 10% of the work."""
    doc = pymupdf.open(PDF_DIR / f"{work}.pdf")
    n = doc.page_count
    k = k or max(1, min(SAMPLE, (n - 1) // 10))
    layer = layer_if_there(ocr_layer(work) / f"{work}.pdf")
    rng = random.Random(seed if seed is not None else work)
    order = list(range(1, n + 1))
    rng.shuffle(order)

    def words(i):
        page = doc[i - 1]
        w = len(page.get_text("words"))
        if w < 20 and layer is not None and layer.page_count == n:
            w = len(layer[i - 1].get_text("words"))
        return w

    # pages with text first (a title page or a blank leaf says nothing about the layout), then any
    picked = [i for i in order if words(i) >= 60][:k]
    picked += [i for i in order if i not in picked][:k - len(picked)]
    return sorted(picked)


def render(work: str, n: int) -> bytes:
    page = pymupdf.open(PDF_DIR / f"{work}.pdf")[n - 1]
    zoom = MAX_PX / max(page.rect.width, page.rect.height)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("jpeg", jpg_quality=85)


def request(work: str, pages: list[int]) -> dict:
    names = standards()
    menu = "\n".join(f"- {k}: {v}" for k, v in names.items())
    text = f"""These are {len(pages)} random pages (scans {', '.join(map(str, pages))}) of one early
modern printed book. A transcriber will read every page of it with a fixed prompt plus one layout
standard. Choose the standard that fits most pages of this book:
{menu}

Rules for the choice:
- gewoon: one column of running prose; margin notes short and few. Choose it whenever it fits.
- twee_kolommen: running text in two columns on most pages; drie_kolommen: three or more.
- parallel: two columns that hold the same text in two languages or versions.
- kanttekeningen: one column with many or long margin notes beside it.
- regels / tabel / beeld: as described; grieks_hebreeuws: Greek or Hebrew on most pages as text
  of its own (whole lines, lists, columns), not a word here and there.
- Two printed pages on one scan are cut apart anyway: judge the printed pages.
Then add modules from {', '.join(MODULES)} that also hold for a good part of the pages (not the
standard you chose, at most two). Then write book_note: at most 70 words, in English, of facts a
transcriber of THIS book needs and the standard does not say: the languages and scripts, what the
margins hold (references, summaries, long notes), recurring elements (verse numbers, numbered
theses, dialogue speakers, line numbers), and anything odd. Facts about the book only, no rules
that change how to transcribe. Set notes_after to true when the margin notes are so many or so long
that a reader should get the main text first and the notes after it.

Answer with JSON only:
{{"standard": "...", "modules": [...], "book_note": "...", "notes_after": false,
  "languages": ["la", ...], "figures": false, "pages": {{"<scan>": "one line on that page's layout", ...}},
  "reason": "one sentence"}}"""
    content = []
    for n in pages:
        content.append({"type": "text", "text": f"Scan {n}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(render(work, n)).decode()}})
    content.append({"type": "text", "text": text})
    return {"model": MODEL, "max_tokens": 2000, "messages": [{"role": "user", "content": content}]}


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in the answer: {text[:200]}")
    d = json.loads(m.group(0))
    names = standards()
    if d.get("standard") not in names:
        raise ValueError(f"unknown standard {d.get('standard')!r}")
    d["modules"] = [x for x in d.get("modules") or [] if x in MODULES and x != d["standard"]][:2]
    d["book_note"] = " ".join(str(d.get("book_note") or "").split())[:600]
    return d


def choose(work: str, client=None, force: bool = False, via: str = "api") -> dict:
    """The work's profile: from layout_profile.json, or chosen now and saved there. via="claude": asked through the
    Claude Code CLI (common/via_claude.py), as transcribe.py --via claude reads; the subscription pays, so it costs 0."""
    out = DATA / work / "layout_profile.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    pages = sample_pages(work)
    req = request(work, pages)
    if via == "claude":
        msg = via_claude.ask("You choose the layout standard of a printed book. Answer with JSON only.",
                             req["messages"][0]["content"], MODEL)
    else:
        msg = (client or client_()).messages.create(**req)
    text = "".join(b.text for b in msg.content if b.type == "text")
    d = parse(text)
    cost = 0 if via == "claude" else (msg.usage.input_tokens * PRICE[0] + msg.usage.output_tokens * PRICE[1]) / 1e6
    d.update({"work": work, "sample": pages, "model": msg.model, "cost_usd": round(cost, 5)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def prompt_for(profile: dict) -> str:
    """The transcription prompt for a work with this profile. A plain work without modules or pictures
    keeps the base prompt as it is, so its pages match the works already done."""
    base = BASE.read_text(encoding="utf-8")
    std, mods = profile["standard"], profile.get("modules") or []
    if std == "gewoon" and not mods and not profile.get("figures"):
        return base
    parts = [base, (LAYOUT / "_common.txt").read_text(encoding="utf-8").strip()]
    parts += [body(x) for x in [std, *mods] if body(x)]
    if profile.get("book_note"):
        parts.append("ABOUT THIS BOOK (facts, not new rules): " + profile["book_note"])
    return "\n\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("works", nargs="+", help="work names (PDF stem) or ids (W00339)")
    ap.add_argument("--force", action="store_true", help="choose again")
    ap.add_argument("--show", action="store_true", help="print the prompt the profile gives")
    ap.add_argument("--via", choices=["api", "claude"], default="api", help="claude: through the Claude Code CLI (subscription)")
    args = ap.parse_args()
    refuse_foreign(DATA)
    client = None
    for w in args.works:
        if re.fullmatch(r"W\d{5}(-\d+)?", w):
            hits = sorted(PDF_DIR.glob(f"{w}_*.pdf"))
            if not hits:
                print(f"{w}: no PDF")
                continue
            w = hits[0].stem
        client = client or (client_() if args.via == "api" else None)
        d = choose(w, client, args.force, via=args.via)
        print(f"{w[:6]}\t{d['standard']}\t{'+'.join(d.get('modules') or []) or '-'}\t"
              f"notes_after={d.get('notes_after')}\t${d.get('cost_usd', 0):.4f}\t{d.get('book_note', '')[:100]}")
        if args.show:
            print(prompt_for(d)[len(BASE.read_text(encoding='utf-8')):])


if __name__ == "__main__":
    main()
