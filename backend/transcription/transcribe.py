"""Page-by-page diplomatic transcription of a scanned book with Claude vision.

Works on a PDF with an OCR text layer. The transcription instructions live in a prompt file
(prompts/early_modern_latin_paragraphs.txt): the model transcribes paragraph by paragraph and
ocr_split.py restores the printed lines from the OCR. A page without a usable OCR layer is
transcribed too: each paragraph stays one line (<page lines="none">, with a warning), and the
page is listed in no_ocr.txt. Once the page has an OCR layer, --resplit cuts it into lines.

Examples
  python transcription/transcribe.py book.pdf --pages 21 --render-only        # check image legibility, no API call
  python transcription/transcribe.py book.pdf --pages 20-24 --compare         # pilot: Haiku vs Sonnet vs Opus side by side
  python transcription/transcribe.py book.pdf --batch                         # whole book, cheap default model, 50% off
  python transcription/transcribe.py book.pdf --retry-failed-with claude-sonnet-5-5
  python transcription/transcribe.py book.pdf --merge-only                    # rebuild book.xml
  python transcription/transcribe.py book.pdf --resplit                       # split the paragraphs into lines again
  python transcription/transcribe.py book.pdf --long-s                        # write ſ as s in the saved pages

Needs ANTHROPIC_API_KEY in the environment (never put the key in this file).
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import dubbel, llm  # noqa: E402
from common.env import saved_env  # noqa: E402
# these stood here; older callers still take them from this file
from common.bookfile import INLINE, book_json, flag_merged_lines, letters, line_markup, mark_numbered  # noqa: E402,F401
from common.llm import PRICES, cost_of  # noqa: E402,F401
from common.pagespec import parse_pages  # noqa: E402,F401
from common.paths import DATA, PROMPTS, refuse_foreign  # noqa: E402
from common.via_claude import Usage  # noqa: E402,F401
from transcription import figures, layout_profile as profile  # noqa: E402

DEFAULT_PROMPT = PROMPTS / "early_modern_latin_paragraphs.txt"


COMPARE_MODELS = [llm.MODEL_HAIKU, "claude-sonnet-5-5", "claude-opus-5-5"]


# ---------------------------------------------------------------- pages

from transcription.pdf_source import IMAGE_EXTS, Book  # noqa: E402,F401  (step 72)
from transcription.page_store import DIAGRAM_MODEL, DIAGRAM_NOTE, tag  # noqa: E402,F401  (step 75)


# ---------------------------------------------------------------- requests

from transcription.api_requests import FALLBACK_MODELS, build_params, call_sync, page_requests, prompt_sha  # noqa: E402,F401  (step 73)


# ---------------------------------------------------------------- output

from transcription.answer import (BARE_AMP, BARE_LT, LOOKALIKES, NO_VALUE, balance_tags, close_lines,  # noqa: E402,F401
                                  escape_notes, join_parts, parse_page)  # (step 74)
from transcription.merge import move_catchwords, trailing_catchword  # noqa: E402,F401  (step 76)
from transcription.page_store import Output  # noqa: E402  (step 75)


# ---------------------------------------------------------------- run modes

from transcription.runners import MAX_BATCH_BYTES, redo_diagrams, run_batch, run_compare, run_sync  # noqa: E402,F401  (step 77)


# ---------------------------------------------------------------- main

def workers_for(via: str, given: int | None) -> int:
    """Parallel requests: as given, else 4, and 2 through the subscription (gentle with its limits unless asked)."""
    return given if given else 2 if via == "claude" else 4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="PDF file with an OCR text layer")
    ap.add_argument("--pages", help="e.g. 1-10,45 (default: all)")
    ap.add_argument("--model", default=llm.MODEL_HAIKU)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT,
                    help="transcription instructions (paragraphs, split into lines by ocr_split.py)")
    ap.add_argument("--out", type=Path, default=DATA, help="output root (a subfolder per book)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max-px", type=int, default=2576, help="long edge of the image sent to the model (its limit; more is scaled down)")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--profile", action="store_true",
                    help="read with the work's layout standard (layout_profile.py: chosen from a sample of pages, saved as layout_profile.json)")
    ap.add_argument("--whole-pages", action="store_true", help="never read a hard page in parts (see parts.py)")
    ap.add_argument("--workers", type=int, help="parallel requests in sync mode (default 4; 2 with --via claude)")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (50%% cheaper, slower)")
    ap.add_argument("--via", choices=["api", "claude"], default="api",
                    help="claude: run on a Claude subscription through the Claude Code CLI (see via_claude.py)")
    ap.add_argument("--poll", type=int, default=60, help="seconds between batch status checks")
    ap.add_argument("--force", action="store_true", help="redo pages that are already transcribed")
    ap.add_argument("--compare", action="store_true", help="pilot: run the pages on several models side by side")
    ap.add_argument("--compare-models", default=",".join(COMPARE_MODELS))
    ap.add_argument("--retry-failed-with", metavar="MODEL", help="redo pages listed in failed.txt with MODEL")
    ap.add_argument("--render-only", action="store_true", help="only write the page images, no API calls")
    ap.add_argument("--redo-diagrams", action="store_true",
                    help="redo the pages listed in diagrams.txt (Haiku's first version of a diagram) with --diagram-model")
    ap.add_argument("--diagram-model", default=DIAGRAM_MODEL, metavar="MODEL", help="the model for --redo-diagrams")
    ap.add_argument("--merge-only", action="store_true", help="only rebuild book.xml")
    ap.add_argument("--resplit", action="store_true",
                    help="split the saved paragraph pages into lines again and rebuild book.xml, no API calls")
    ap.add_argument("--long-s", action="store_true",
                    help="write every ſ in the saved pages as s and rebuild book.xml, no API calls")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-check failed pages (.xml.bad) with the current rules, no API calls")
    ap.add_argument("--ook-dubbel", action="store_true",
                    help="transcribe even if the work is held as a probable duplicate (common/dubbel.py)")
    args = ap.parse_args()
    args.workers = workers_for(args.via, args.workers)
    refuse_foreign(args.out)

    # A probable duplicate of a work in the library waits for a decision (backend/docs/DUBBELEN.md); a work that is
    # already transcribed is never held, so its repairs go on. Only runs that call the API are held.
    free = args.merge_only or args.resplit or args.long_s or args.revalidate or args.render_only
    if not free and not args.ook_dubbel and not (args.out / args.source.stem / "book.json").exists() \
            and (why := dubbel.held(args.source.stem)):
        sys.exit(f"wacht, waarschijnlijk dubbel: {why}. Na een besluit de regel weghalen, of --ook-dubbel.")
    book = Book(args.source)
    out = Output(args.out / args.source.stem, book)
    print(f"{args.source.name}: {book.count} pages -> {out.root}")

    if args.merge_only:
        out.merge(book)
        return
    if args.resplit:
        out.resplit()
        out.merge(book)
        return
    if args.long_s:
        out.long_s()
        out.merge(book)
        return
    if args.revalidate:
        out.revalidate()
        out.write_failed()
        out.merge(book)
        return

    pages = parse_pages(args.pages, book.count)
    if args.render_only:
        d = out.root / "images"
        d.mkdir(exist_ok=True)
        for n in pages:
            (d / f"{tag(n)}.jpg").write_bytes(book.render_jpeg(n, args.dpi, args.max_px))
        print(f"Wrote {len(pages)} image(s) to {d}")
        return

    key = saved_env("ANTHROPIC_API_KEY")
    workspace = saved_env("ANTHROPIC_WORKSPACE_ID")
    if args.via == "claude":
        figures.VIA = "claude"
        if args.batch or args.compare:
            sys.exit("--batch and --compare need the API; leave them out with --via claude.")
        key = key or "subscription"  # not used: via_claude.py strips the key from the CLI's environment
    if not key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        sys.exit("ANTHROPIC_API_KEY is not set in this terminal. After `setx`, open a NEW terminal "
                 "(setx does not change the current one), or for this terminal only run:\n"
                 '  $env:ANTHROPIC_API_KEY = "sk-ant-api03-..."   (PowerShell)')
    if key.endswith("..."):
        sys.exit("ANTHROPIC_API_KEY is still the placeholder \"sk-ant-...\". Set your real key "
                 "from https://console.anthropic.com/settings/keys.")

    prompt = args.prompt.read_text(encoding="utf-8")
    if args.profile:
        if args.model != profile.MODEL:
            sys.exit(f"--profile reads with {profile.MODEL} only")
        prof = profile.choose(args.source.stem, None if args.via == "claude" else profile.client_(), via=args.via)
        prompt = profile.prompt_for(prof)
        print(f"Layout standard: {prof['standard']}{' + ' + '+'.join(prof['modules']) if prof.get('modules') else ''}"
              f" (prompt {prompt_sha(prompt)}; from scans {prof.get('sample')})")
    client = llm.client(workspace, max_retries=5)
    model = args.model

    try:
        if args.compare:
            run_compare(client, args, book, prompt, out, pages)
            return
        if args.retry_failed_with:
            model = args.retry_failed_with
            pages = [n for n in out.read_failed() if n in set(pages)]
            if not pages:
                print("No failed pages to retry.")
                return
        elif args.redo_diagrams:
            redo_diagrams(client, args, book, prompt, out, pages)
            pages = []
        elif not args.force:
            # A page marked as missing (a failed page, or one skipped earlier) is tried again.
            pages = [n for n in pages if not out.transcribed(n)]
        if args.redo_diagrams:
            pass
        elif not pages:
            print("Nothing to do: all selected pages are already transcribed.")
        else:
            print(f"Retrying {len(pages)} failed page(s)" if args.retry_failed_with else
                  f"Transcribing {len(pages)} page(s) with {model}, effort {args.effort}"
                  f"{', batch' if args.batch else ''}")
            if args.batch:
                run_batch(client, args, book, prompt, out, pages, model)
            else:
                run_sync(client, args, book, prompt, out, pages, model)
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: set ANTHROPIC_API_KEY to a valid key.")
    except anthropic.BadRequestError as e:
        if "workspace" not in str(e.message):
            raise
        sys.exit("This API key is not tied to a workspace. Either create a key inside a workspace "
                 "(console.anthropic.com > Settings > Workspaces), or save the workspace ID:\n"
                 '  setx ANTHROPIC_WORKSPACE_ID "wrkspc_..."')
    finally:
        if not args.compare:
            out.write_failed()
        print(f"Spent this run: ${out.spent:.2f}")

    if not args.compare:
        out.merge(book)


if __name__ == "__main__":
    main()
