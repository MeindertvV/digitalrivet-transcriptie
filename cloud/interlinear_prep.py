"""Steps 1 and 2 of the interlinear text for the works of interlinear_works.tsv (no model, no credit).

  python cloud/interlinear_prep.py --shard 0 --of 4 [--base https://.../pdfs] [--limit 1]

The works already translated on the server: for each, its units (id, start, heading, Latin, English: the
server's translation_en.json, nothing else) are fetched from <base>/interlinear_input/<work>.json.gz, and
backend/sentences/interlinear.py --no-haiku pairs the sentences with Bertalign/LaBSE and looks the words up
in Whitaker's Words. Out: interlinear_out/<work>/align_cache.json and prep.json (what Haiku would get),
committed and pushed to the branch interlinear-<shard> after each work. A work already in interlinear_out/
is skipped, so a stopped run just starts again. New works get the same as step 6 of cloud/run.py.
Needs requirements.txt and requirements-interlinear.txt; LaBSE (~1.9 GB) comes on the first run.
"""
import argparse, gzip, os, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "interlinear_out"
WORK = Path(os.environ.get("INTERLINEAR_WORK", "/tmp/interlinear_work"))
DEFAULT_BASE = os.environ.get("PDF_BASE", "")


def sh(*cmd, check=True):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True)


def push(work: str, branch: str) -> None:
    sh("git", "add", "-f", f"interlinear_out/{work}")
    if sh("git", "diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    sh("git", "commit", "-q", "-m", f"interlinear steps 1-2 {work}")
    for _ in range(5):
        if sh("git", "push", "-q", "origin", f"HEAD:{branch}", check=False).returncode == 0:
            return
        sh("git", "pull", "-q", "--rebase", "origin", branch, check=False)
        time.sleep(10)
    print(f"push failed for {work}; it stays committed here", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--base", default=DEFAULT_BASE, help="URL of the folder with the PDFs (and interlinear_input/)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many works (0: all)")
    ap.add_argument("--no-push", action="store_true", help="a trial: commit nothing")
    a = ap.parse_args()
    if not a.base:
        sys.exit("give --base or PDF_BASE")
    branch = f"interlinear-{a.shard}"
    if not a.no_push:
        sh("git", "fetch", "-q", "origin", branch, check=False)
        if sh("git", "rev-parse", "--verify", "-q", f"origin/{branch}", check=False).returncode == 0:
            sh("git", "checkout", "-q", "-B", branch, f"origin/{branch}")
        else:
            sh("git", "checkout", "-q", "-B", branch)
    rows = [l.split("\t") for l in (ROOT / "interlinear_works.tsv").read_text().splitlines() if "\t" in l]
    mine = [(w, int(n)) for i, (w, n) in enumerate(rows) if i % a.of == a.shard]
    todo = [(w, n) for w, n in mine if not (OUT / w / "prep.json").exists()]
    print(f"shard {a.shard}/{a.of}: {len(todo)} of {len(mine)} works left, {sum(n for _, n in todo)} Latin words", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.pop("ANTHROPIC_API_KEY", None)          # nothing here asks a model
    done = failed = 0
    for w, n in todo:
        t0 = time.time()
        src = WORK / f"{w}.json"
        try:
            req = urllib.request.Request(f"{a.base.rstrip('/')}/interlinear_input/{w}.json.gz",
                                         headers={"User-Agent": "digitalrivet-transcriptie/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                src.write_bytes(gzip.decompress(r.read()))
        except Exception as e:
            print(f"{w[:6]}: download failed: {e}", flush=True)
            failed += 1
            continue
        out = OUT / w
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "prep.log", "w") as fh:
            rc = subprocess.run([sys.executable, "sentences/interlinear.py", "--src", str(src), "--out", str(out),
                                 "--no-haiku"], cwd=ROOT / "backend", env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        src.unlink(missing_ok=True)
        ok = (out / "prep.json").exists()
        print(f"{time.strftime('%H:%M:%S')} {w[:6]}: rc={rc} prep.json={ok} {n} words, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if not ok:
            failed += 1
        if not a.no_push:
            push(w, branch)
        done += 1
        if a.limit and done >= a.limit:
            break
    print(f"done: {done} works, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
