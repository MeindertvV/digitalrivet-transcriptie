"""OCR layers in a cloud session: Tesseract (no model, no credit) for the works of ocr_works.tsv.

  python cloud/ocr.py --shard 0 --of 4 [--workers N] [--base https://.../pdfs] [--limit 1]

The same layer as DigitalRivet's ops/night/ocr_layers_pc.py (keep scan_dpi and ocr_page as they are there):
each page at its scan's own resolution, Latin (cloud/tessdata/lat.traineddata, the server's file), only the
invisible text kept. Each shard takes every N-th work of ocr_works.tsv, downloads its PDF, OCRs it in parts of
20 pages over --workers processes, checks that the layer has the PDF's page count, and commits and pushes
ocr_out/<work>.pdf to the branch ocr-<shard>. A work already in ocr_out/ is skipped, so a stopped run just
starts again. The server (testset/work/cloud_ocr_import.sh) copies the layers into its OCR inbox.
"""
import argparse, os, re, shutil, subprocess, sys, time, urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ocr_out"
TESSDATA = str(ROOT / "cloud" / "tessdata")
WORK = Path(os.environ.get("OCR_WORK", "/tmp/ocr_work"))
DEFAULT_BASE = os.environ.get("PDF_BASE", "")
PART = 20
DEFAULT_DPI = 300
IMAGE_DO = re.compile(rb"q\s+[\d.\s-]+cm\s+/I\d+\s+Do\s+Q")


# --- as ops/night/ocr_layers_pc.py (scan_dpi, ocr_page): change them together ---
def scan_dpi(page):
    imgs = [i for i in page.get_image_info() if i["bbox"][2] > i["bbox"][0]]
    if not imgs:
        return DEFAULT_DPI
    i = max(imgs, key=lambda i: (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]))
    return min(600, max(72, round(i["width"] / ((i["bbox"][2] - i["bbox"][0]) / 72))))


def ocr_page(page, out, tessdata):
    rot = page.rotation
    page.set_rotation(0)
    layer = out.new_page(width=page.rect.width, height=page.rect.height)
    pix = page.get_pixmap(dpi=scan_dpi(page), colorspace=pymupdf.csRGB, alpha=False)
    ocr = pymupdf.open("pdf", pix.pdfocr_tobytes(compress=False, language="lat", tessdata=tessdata))
    op = ocr[0]
    words = len(op.get_text("words"))
    if words:
        xref = op.get_contents()[0]
        ocr.update_stream(xref, IMAGE_DO.sub(b"", ocr.xref_stream(xref)), compress=True)
        for img in op.get_images():
            ocr.update_stream(img[0], b"", compress=True)
            ocr.xref_set_key(op.xref, "Resources/XObject/" + img[7], "null")
        layer.show_pdf_page(layer.rect, ocr, 0, keep_proportion=False)
    layer.set_rotation(rot)
    return words
# ---


def ocr_part(job):
    pdf, a, b, out = job
    if not os.path.exists(out):
        with pymupdf.open(pdf) as d, pymupdf.open() as layer:
            for i in range(a, b):
                ocr_page(d[i], layer, TESSDATA)
            layer.save(out + ".tmp", garbage=3, deflate=True)
        os.replace(out + ".tmp", out)
    return out


def sh(*cmd, check=True):
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, capture_output=True)


def push(name: str, branch: str) -> None:
    sh("git", "add", "-f", f"ocr_out/{name}")
    if sh("git", "diff", "--cached", "--quiet", check=False).returncode == 0:
        return
    sh("git", "commit", "-q", "-m", f"OCR layer {name}")
    for _ in range(5):
        if sh("git", "push", "-q", "origin", f"HEAD:{branch}", check=False).returncode == 0:
            return
        sh("git", "pull", "-q", "--rebase", "origin", branch, check=False)
        time.sleep(10)
    print(f"push failed for {name}; it stays committed here", flush=True)


def download(base: str, name: str, dest: Path) -> None:
    # Cloudflare refuses Python's own User-Agent (403), so name ourselves
    req = urllib.request.Request(f"{base.rstrip('/')}/{name}", headers={"User-Agent": "digitalrivet-transcriptie/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(str(dest) + ".tmp", "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    os.replace(str(dest) + ".tmp", dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--base", default=DEFAULT_BASE, help="URL of the folder with the PDFs")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many works (0: all)")
    ap.add_argument("--no-push", action="store_true", help="a trial: commit nothing")
    a = ap.parse_args()
    if not a.base:
        sys.exit("give --base or PDF_BASE")
    if not Path(TESSDATA, "lat.traineddata").is_file():
        sys.exit(f"no lat.traineddata in {TESSDATA}")
    branch = f"ocr-{a.shard}"
    if not a.no_push:
        sh("git", "fetch", "-q", "origin", branch, check=False)
        if sh("git", "rev-parse", "--verify", "-q", f"origin/{branch}", check=False).returncode == 0:
            sh("git", "checkout", "-q", "-B", branch, f"origin/{branch}")
        else:
            sh("git", "checkout", "-q", "-B", branch)
    rows = [l.split("\t") for l in (ROOT / "ocr_works.tsv").read_text().splitlines() if "\t" in l]
    mine = [(n, int(p)) for i, (n, p) in enumerate(rows) if i % a.of == a.shard]
    todo = [(n, p) for n, p in mine if not (OUT / n).exists()]
    print(f"shard {a.shard}/{a.of}: {len(todo)} of {len(mine)} works left, {sum(p for _, p in todo)} pages; "
          f"{a.workers} processes", flush=True)
    OUT.mkdir(exist_ok=True)
    done = failed = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for name, _ in todo:
            t0 = time.time()
            pdf = WORK / "pdfs" / name
            parts_dir = WORK / "parts" / name
            pdf.parent.mkdir(parents=True, exist_ok=True)
            parts_dir.mkdir(parents=True, exist_ok=True)
            try:
                if not pdf.is_file():
                    download(a.base, name, pdf)
                with pymupdf.open(pdf) as d:
                    pages = d.page_count
                jobs = [(str(pdf), s, min(s + PART, pages), str(parts_dir / f"{s:05d}.pdf"))
                        for s in range(0, pages, PART)]
                parts = list(ex.map(ocr_part, jobs))
                with pymupdf.open() as out:
                    for p in parts:
                        with pymupdf.open(p) as src:
                            out.insert_pdf(src)
                    words = sum(len(pg.get_text("words")) for pg in out)
                    if out.page_count != pages:
                        raise RuntimeError(f"layer has {out.page_count} pages, the PDF {pages}")
                    out.save(str(OUT / name) + ".tmp", garbage=3, deflate=True)
                os.replace(str(OUT / name) + ".tmp", OUT / name)
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} {name[:6]}: failed: {e}", flush=True)
                failed += 1
                continue
            shutil.rmtree(parts_dir, ignore_errors=True)
            pdf.unlink(missing_ok=True)
            secs = time.time() - t0
            print(f"{time.strftime('%H:%M:%S')} {name[:6]}: {pages} p., {words} words, {secs/60:.1f} min "
                  f"({secs/pages:.1f} s/p.)", flush=True)
            if not a.no_push:
                push(name, branch)
            done += 1
            if a.limit and done >= a.limit:
                break
    print(f"done: {done} works, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
