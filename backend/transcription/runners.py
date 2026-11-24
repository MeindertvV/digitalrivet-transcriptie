"""The ways a transcription runs (REFACTOR_PLAN.md step 77: was the run modes of transcribe.py, which still has the
names).

run_sync asks page by page, a few at a time (directly, or through the Claude Code CLI with --via claude);
run_batch sends the pages as message batches and collects them, also on a later run (batches.json); run_compare
reads the same pages with several models side by side (compare/compare.html); redo_diagrams does the pages in
diagrams.txt again with a better model.
"""

import html
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import via_claude  # noqa: E402
from transcription.answer import join_parts  # noqa: E402
from transcription.api_requests import call_sync, page_requests, prompt_sha  # noqa: E402
from transcription.page_store import tag  # noqa: E402

MAX_BATCH_BYTES = 200 * 1024 * 1024  # API limit is 256 MB per batch


def run_sync(client, args, book, prompt, out, pages, model, pages_dir=None):
    def work(n):
        try:
            msgs = []
            for params, _ in page_requests(args, book, prompt, n, model):
                if args.via == "claude":
                    msgs.append(via_claude.ask(prompt, params["messages"][0]["content"], model, args.effort))
                else:
                    msgs.append(call_sync(client, params))
            msg = msgs[0] if len(msgs) == 1 else join_parts(msgs, model)
            out.save(n, msg, model, batch=False, pages_dir=pages_dir, prompt=prompt_sha(prompt))
        except anthropic.AuthenticationError:
            raise
        except anthropic.BadRequestError as e:
            if "workspace" in str(e.message):
                raise  # same for every page: stop the run
            out.fail(n, f"bad request: {e.message}", pages_dir)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            out.fail(n, f"API error (retry later): {e}", pages_dir)
        except (RuntimeError, subprocess.TimeoutExpired) as e:  # --via claude
            out.fail(n, f"{e} (retry later)", pages_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, n) for n in pages]
        for f in as_completed(futures):
            f.result()  # re-raises AuthenticationError

def redo_diagrams(client, args, book, prompt, out, pages):
    """--redo-diagrams: the pages listed in diagrams.txt (as far as selected), again with args.diagram_model.
    Nothing starts this by itself: the list is for the go-ahead."""
    path = out.root / "diagrams.txt"
    listed = {int(l.split("\t")[0]) for l in path.read_text(encoding="utf-8").splitlines() if l[:1].isdigit()} \
        if path.exists() else set()
    again = [n for n in pages if n in listed]
    if not again:
        print("No pages with a diagram to redo (diagrams.txt is empty or not in the selection).")
        return
    print(f"{len(again)} page(s) with a diagram again with {args.diagram_model}: {','.join(map(str, again))}")
    args.force = True   # the pages are already transcribed
    run_sync(client, args, book, prompt, out, again, args.diagram_model)


def run_batch(client, args, book, prompt, out, pages, model):
    state_file = out.root / "batches.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else []
    wanted = set(pages)

    def stale(b) -> bool:
        """A batch from an earlier run of which no page is left to do: collecting it again would
        only wait for a batch that no longer matters (a vanished request kept the loop going)."""
        return not args.force and not any(not out.transcribed(n) and n in wanted for n in b["pages"])

    keep = []
    for b in state:
        if stale(b):
            print(f"Batch {b['id']} of an earlier run is done; not collected again")
        else:
            keep.append(b)
    state = keep
    if state:
        state_file.write_text(json.dumps(state, indent=2))
    else:
        state_file.unlink(missing_ok=True)

    # Submit new batches (unless a previous run's batches are still pending for these pages).
    pending_pages = {n for b in state for n in b["pages"]}
    todo = [n for n in pages if n not in pending_pages]
    chunk, size = [], 0
    shas = {}  # page -> prompt_sha, kept in the state file: results may come in on a later run

    def submit():
        nonlocal chunk, size
        if not chunk:
            return
        batch = client.messages.batches.create(requests=[r for _, r in chunk])
        state.append({"id": batch.id, "model": model, "pages": sorted({n for n, _ in chunk}),
                      "prompts": {str(n): shas[n] for n, _ in chunk}})
        state_file.write_text(json.dumps(state, indent=2))
        print(f"Submitted batch {batch.id} with {len(chunk)} request(s)")
        chunk, size = [], 0

    for n in todo:
        requests = page_requests(args, book, prompt, n, model)
        shas[n] = prompt_sha(prompt)
        # the parts of a page stay in one batch: its answers are joined when the batch is collected
        req_size = sum(b * 4 // 3 + len(prompt.encode()) + 1024 for _, b in requests)
        if size + req_size > MAX_BATCH_BYTES:
            submit()
        for k, (params, _) in enumerate(requests, 1):
            custom = tag(n) if len(requests) == 1 else f"{tag(n)}-{k}"
            chunk.append((n, Request(custom_id=custom, params=MessageCreateParamsNonStreaming(**params))))
        size += req_size
    submit()

    # Poll and collect every batch in the state file. A batch whose pages are all transcribed by
    # then (perhaps by an earlier run) is dropped rather than collected again.
    while state:
        for b in list(state):
            info = client.messages.batches.retrieve(b["id"])
            if info.processing_status != "ended":
                c = info.request_counts
                print(f"Batch {b['id']}: {c.processing} processing, {c.succeeded} done, {c.errored} errored")
                continue
            answers: dict[int, dict[int, object]] = {}   # page -> its answers, part by part
            for res in client.messages.batches.results(b["id"]):
                n, _, k = res.custom_id[1:].partition("-")
                answers.setdefault(int(n), {})[int(k or 0)] = res.result
            for n, rs in answers.items():
                if out.transcribed(n) and not args.force:
                    continue   # an earlier run transcribed this page already
                bad = next((r for r in rs.values() if r.type != "succeeded"), None)
                if bad is None:
                    msgs = [rs[k].message for k in sorted(rs)]
                    msg = msgs[0] if len(msgs) == 1 else join_parts(msgs, b["model"])
                    out.save(n, msg, b["model"], batch=True, prompt=b.get("prompts", {}).get(str(n)))
                elif bad.type == "errored":
                    out.fail(n, f"batch error: {bad.error.error.type if hasattr(bad.error, 'error') else bad.error}")
                else:
                    out.fail(n, f"batch {bad.type}")
            state.remove(b)
            state_file.write_text(json.dumps(state, indent=2))
        if state:
            time.sleep(args.poll)
    state_file.unlink(missing_ok=True)


def run_compare(client, args, book, prompt, out, pages):
    cmp_root = out.root / "compare"
    img_dir = cmp_root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    models = args.compare_models.split(",")
    spend = {}
    for model in models:
        print(f"--- {model}")
        before = out.spent
        pages_dir = cmp_root / model / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        run_sync(client, args, book, prompt, out, pages, model, pages_dir=pages_dir)
        spend[model] = out.spent - before
    out.failed.clear()  # compare failures are visible in the HTML, not in failed.txt

    rows = []
    for n in pages:
        (img_dir / f"{tag(n)}.jpg").write_bytes(book.render_jpeg(n, args.dpi, args.max_px))
        cells = [f'<td><img src="img/{tag(n)}.jpg"></td>']
        for model in models:
            d = cmp_root / model / "pages"
            f = d / f"{tag(n)}.xml"
            text = f.read_text(encoding="utf-8") if f.exists() else \
                "FAILED\n\n" + ((d / f"{tag(n)}.xml.bad").read_text(encoding="utf-8")
                                if (d / f"{tag(n)}.xml.bad").exists() else "")
            cells.append(f"<td><pre>{html.escape(text)}</pre></td>")
        rows.append(f"<tr><th colspan={len(models) + 1}>Page {n}</th></tr><tr>{''.join(cells)}</tr>")
    head = "".join(f"<th>{m}<br>${spend[m]:.3f} for {len(pages)} page(s), "
                   f"≈ ${spend[m] / len(pages) * book.count:.2f} for the book "
                   f"(½ with --batch)</th>" for m in models)
    page = f"""<!doctype html><meta charset="utf-8"><title>Model comparison</title>
<style>body{{font-family:sans-serif;margin:16px}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;vertical-align:top;padding:6px}} img{{width:100%;max-width:700px}}
pre{{white-space:pre-wrap;font-family:"Junicode","Cambria",serif;font-size:14px}} td{{width:{100 // (len(models) + 1)}%}}</style>
<h1>{html.escape(book.source.name)} — effort {args.effort}, {args.max_px}px</h1>
<table><tr><th>Scan</th>{head}</tr>{''.join(rows)}</table>"""
    (cmp_root / "compare.html").write_text(page, encoding="utf-8")
    print(f"Comparison written to {cmp_root / 'compare.html'}")
