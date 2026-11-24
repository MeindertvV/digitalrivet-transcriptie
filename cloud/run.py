"""Transcribe the works of works.tsv in a cloud session: Haiku only, through `claude -p` (--via claude).

  python cloud/run.py --shard 0 --of 4 [--workers 4] [--base https://.../pdfs]

Each shard takes every N-th work of works.tsv (from the top), downloads its PDF and runs the steps after
one another into out/<work>/, all with Haiku on the subscription (claude -p):
  1. transcribe.py --via claude (pictures are cut as part of it)
  2. the pages in failed.txt once more
  3. ocr_check/line_compare.py: every line beside the scan's OCR (ocr_regels.json; no model)
  4. contents/toc_context.py: the contents (toc_context.json)
  5. translation/translate.py --lang en --via claude (translation_en.json)
Then it deletes the PDF, and commits and pushes out/<work>/ to the branch results-<shard> (also after step 1,
so a stop loses little). A step that fails is logged in out/<work>/<step>.log; the next steps still run. A work whose out/<work>/book.json is already there is skipped,
so a stopped run just starts again. Stops when the subscription's usage limit is reached.
"""
import argparse, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
PDFS = Path(os.environ.get("PDF_CACHE", "/tmp/pdfs"))
DEFAULT_BASE = os.environ.get("PDF_BASE", "")
DONE = ("book.json", "ocr_regels.json", "toc_context.json", "translation_en.json")   # a work with all of these is skipped


def sh(*cmd, check=True):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True)


def push(work: str, branch: str) -> None:
    sh("git", "add", "-f", f"out/{work}")
    if sh("git", "diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    sh("git", "commit", "-q", "-m", f"transcribed {work}")
    for _ in range(5):
        if sh("git", "push", "-q", "origin", f"HEAD:{branch}", check=False).returncode == 0:
            return
        sh("git", "pull", "-q", "--rebase", "origin", branch, check=False)
        time.sleep(10)
    print(f"push failed for {work}; it stays committed here", flush=True)


def step(work: str, name: str, cmd: list[str], env: dict) -> int:
    """One step for one work, from backend/, its output in out/<work>/<name>.log."""
    log = OUT / work / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        return subprocess.run([sys.executable, *cmd], cwd=ROOT / "backend", env=env,
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default=DEFAULT_BASE, help="URL of the folder with the PDFs")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many works (0: all)")
    a = ap.parse_args()
    if not a.base:
        sys.exit("give --base or PDF_BASE")
    branch = f"results-{a.shard}"
    sh("git", "fetch", "-q", "origin", branch, check=False)
    if sh("git", "rev-parse", "--verify", "-q", f"origin/{branch}", check=False).returncode == 0:
        sh("git", "checkout", "-q", "-B", branch, f"origin/{branch}")
    else:
        sh("git", "checkout", "-q", "-B", branch)
    works = [l.split("\t")[0] for l in (ROOT / "works.tsv").read_text().splitlines() if l.strip()]
    mine = [w for i, w in enumerate(works) if i % a.of == a.shard]
    PDFS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PRDL_DATA": str(OUT), "PYTHONIOENCODING": "utf-8"}
    env.pop("ANTHROPIC_API_KEY", None)          # the subscription, never the API
    done = 0
    for w in mine:
        if all((OUT / w / f).exists() for f in DONE):
            continue
        pdf = PDFS / f"{w}.pdf"
        t0 = time.time()
        try:
            # Cloudflare refuses Python's own User-Agent (403), so name ourselves
            req = urllib.request.Request(f"{a.base.rstrip('/')}/{w}.pdf",
                                         headers={"User-Agent": "digitalrivet-transcriptie/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(pdf, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
        except Exception as e:
            print(f"download failed {w}: {e}", flush=True)
            continue
        wid = w.split("_")[0]
        rc = step(w, "transcribe", ["transcription/transcribe.py", str(pdf), "--via", "claude", "--workers", str(a.workers)], env)
        push(w, branch)
        failed = OUT / w / "failed.txt"
        if failed.exists() and failed.read_text().strip():
            pages = ",".join(l.split("\t")[0] for l in failed.read_text().splitlines() if l.strip())
            step(w, "repair", ["transcription/transcribe.py", str(pdf), "--via", "claude", "--pages", pages,
                               "--workers", str(a.workers)], env)
        if (OUT / w / "book.json").exists():
            if not (OUT / w / "ocr_regels.json").exists():
                step(w, "ocr_regels", ["ocr_check/line_compare.py", str(OUT / w), "--pdf", str(pdf)], env)
            if not (OUT / w / "toc_context.json").exists():
                step(w, "toc_context", ["contents/toc_context.py", wid], env)
            if not (OUT / w / "translation_en.json").exists():
                step(w, "translate", ["translation/translate.py", str(OUT / w), "--lang", "en", "--via", "claude",
                                      "--workers", str(a.workers)], env)
        pdf.unlink(missing_ok=True)
        text = "".join((OUT / w / f).read_text(errors="replace") for f in
                       ("transcribe.log", "repair.log", "toc_context.log", "translate.log") if (OUT / w / f).exists())
        ok = (OUT / w / "book.json").exists()
        print(f"{time.strftime('%H:%M:%S')} {w} rc={rc} book.json={ok} "
              f"en={(OUT / w / 'translation_en.json').exists()} {(time.time()-t0)/60:.1f} min", flush=True)
        push(w, branch)
        done += 1
        if "UsageLimit" in text or "usage limit" in text.lower():
            print("usage limit reached: stopping", flush=True)
            break
        if a.limit and done >= a.limit:
            break


if __name__ == "__main__":
    main()
