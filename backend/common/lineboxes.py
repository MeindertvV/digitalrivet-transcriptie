"""Where every transcribed line sits on its scan: a box per line, from the PDF's own text layer.

The downloaded PDFs (Internet Archive) carry a hidden OCR layer with a position for every word.
That OCR is poor (long s read as f, letters run together), so it cannot be searched for the
transcription; but it is good enough to tell which of its lines is which line of the
transcription. Per scan the OCR lines are put in reading order and paired with the transcribed
main-text lines in one pass (as a diff pairs lines): pairs must keep their order, OCR lines may be
left over (running head, catchword, margin notes) and so may transcribed lines (where the OCR
missed or merged a line). A pair counts when enough of its letters agree.

build_reader.py uses boxes() for the reader, which marks the line under the mouse on the scan.
Run on its own it reports how many lines got a box:

  python lineboxes.py ../backend/data/W01206_Calvin_…_1561 [--pages 190-210] [--show 200]
"""

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/
from common.works import text_pdf as layer_or_pdf  # noqa: E402

HERE = Path(__file__).resolve().parent
MIN_SCORE = 0.55   # share of letters that must agree for a pair to count
ROUGH = 0.45       # below this share of shared letter pairs a pair is not looked at closely
SKIP = 0.25        # what leaving a line unpaired costs in the pairing (pairs scoring below it are not worth it)


from common.text import pairing_letters as norm  # noqa: E402,F401  (step 57)


def grams(s: str) -> Counter:
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def rough(a: Counter, b: Counter, la: int, lb: int) -> float:
    """Shared letter pairs: quick, and never far below similarity() for lines that belong together."""
    return 2 * sum((a & b).values()) / (la + lb - 2) if la > 1 and lb > 1 else 0.0


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False)
    return 2 * sum(bl.size for bl in m.get_matching_blocks()) / (len(a) + len(b))


def gutters(lines, width: float) -> list[float]:
    """The x of every gap between columns of text: strips in the middle of the page that far fewer
    lines have a word across than the text around them (OCR runs a few words over the gap)."""
    full = [ws for ws in lines if len(ws) > 2]
    if len(full) < 10:
        return []
    cover = [0] * (int(width) + 1)
    for ws in full:
        hit = set()
        for x0, _, x1, *_ in ws:
            hit.update(range(max(int(x0), 0), min(int(x1), int(width)) + 1))
        for x in hit:
            cover[x] += 1
    text = sorted(c for c in cover if c)
    low = text[len(text) // 2] * 0.35
    out, run = [], []
    for x in range(int(width * 0.2), int(width * 0.8)):
        if cover[x] < low:
            run.append(x)
        elif run:
            out.append(min(run, key=lambda x: cover[x]) + 0.5)
            run = []
    return out


def column_lines(lines, width: float):
    """One column's OCR lines in reading order: [(text, text without the margin, box)], words in the
    margin (left or right of the column) left out of the box and the second text."""
    starts = sorted(ws[0][0] for ws in lines if len(ws) > 2)
    ends = sorted(ws[-1][2] for ws in lines if len(ws) > 2)
    left = starts[len(starts) // 2] if starts else 0
    right = ends[len(ends) // 2] if ends else width
    w = right - left
    out = []
    for ws in lines:
        inner = [x for x in ws if x[2] > left - w * 0.03 and x[0] < right + w * 0.03] or ws
        box = [min(x[0] for x in inner), min(x[1] for x in inner), max(x[2] for x in inner), max(x[3] for x in inner)]
        out.append((" ".join(x[4] for x in ws), " ".join(x[4] for x in inner), box))
    out.sort(key=lambda o: ((o[2][1] + o[2][3]) / 2, o[2][0]))
    return out


# --- OCR words to lines. The copies that group a page's words per OCR line, each as its caller had it (step 65);
# ocr_lines() below and layout_check.page_side do it inside a longer function.

def word_rows(page) -> list[list]:
    # was transcription/parts.ocr_rows
    lines = {}
    for w in page.get_text("words"):
        lines.setdefault((w[5], w[6]), []).append(w)
    return [sorted(ws) for ws in lines.values()]


# was ocr/ocr_naar_xml.WATERMARK
WATERMARK = re.compile(r"digiti[sz]ed\s+by|google|^\W*o+gle\W*$|internet\s*archive|bayerische\s+staats", re.I)


def rows_without_watermark(page) -> list[list]:
    """The OCR words of a page grouped per OCR line, watermark words left out. Was ocr/ocr_naar_xml.page_words."""
    lines = {}
    for x0, y0, x1, y1, w, b, l, _ in page.get_text("words"):
        lines.setdefault((b, l), []).append((x0, y0, x1, y1, w))
    out = []
    for ws in lines.values():
        if WATERMARK.search(" ".join(w[4] for w in ws)):
            continue
        out.append(sorted(ws))
    return out


def ocr_lines(page):
    """The OCR lines of a pymupdf page in reading order (column_lines); on a page in columns (an
    index) column after column, each line cut at the gaps between them."""
    lines = {}
    for x0, y0, x1, y1, w, b, l, _ in page.get_text("words"):
        lines.setdefault((b, l), []).append((x0, y0, x1, y1, w))
    lines = [sorted(ws) for ws in lines.values()]
    if not lines:
        return []
    width = page.rect.width
    gx = gutters(lines, width)
    cols = [[] for _ in range(len(gx) + 1)]
    for ws in lines:
        parts = {}
        for x in ws:
            parts.setdefault(sum((x[0] + x[2]) / 2 > g for g in gx), []).append(x)
        for c, part in parts.items():
            cols[c].append(part)
    return [o for col in cols if col for o in column_lines(col, width)]


def spans(t: str, med: float) -> range:
    """How many OCR lines a transcribed line may stand for: one, or for a line that holds several
    printed lines (transcribe.py marks those merged; an index entry that runs over two lines) about
    as many as its length says."""
    r = len(t) / med if med else 1
    if r < 1.25:
        return range(1, 2)
    return range(max(1, round(r) - 3), min(round(r) + 4, 80))


def pair(trans: list[str], ocr: list[str]):
    """Pair transcribed lines with OCR lines in order (both normalised): [(i, j, k, score)], line i
    with OCR lines j to j+k. A pair is worth its score less SKIP, times the length of the line in
    OCR lines: a merged line weighs as much as the lines it holds, and taking in an OCR line more
    than it needs only lowers its score."""
    n, m = len(trans), len(ocr)
    lens = sorted(len(o) for o in ocr if len(o) > 10)
    med = lens[len(lens) // 2] if lens else 0
    ks = [spans(t, med) for t in trans]
    wt = [max(1.0, len(t) / med) if med else 1.0 for t in trans]
    tg, og = [grams(t) for t in trans], [grams(o) for o in ocr]
    sim = {}
    def gain(i, j, k, g, lg):
        # the exact score only where the quick one says it could be a pair
        if (i, j, k) not in sim:
            r = rough(tg[i], g, len(trans[i]), lg)
            sim[i, j, k] = (r if len(trans[i]) > 300 else similarity(trans[i], "".join(ocr[j:j + k]))) if r >= ROUGH else r * 0.5
        return wt[i] * (sim[i, j, k] - SKIP)
    # best[i][j]: the best total for trans[i:] and ocr[j:]
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            b, mv = best[i + 1][j], ("t",)
            if best[i][j + 1] > b:
                b, mv = best[i][j + 1], ("o",)
            g, lg = Counter(), 0
            for k in range(1, ks[i].stop):
                if j + k > m:
                    break
                g += og[j + k - 1]
                lg += len(ocr[j + k - 1])
                if k < ks[i].start:
                    continue
                v = gain(i, j, k, g, lg) + best[i + 1][j + k]
                if v > b:
                    b, mv = v, ("p", k)
            best[i][j], move[i][j] = b, mv
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        mv = move[i][j]
        if mv[0] == "p":
            pairs.append((i, j, mv[1], sim[i, j, mv[1]]))
            i, j = i + 1, j + mv[1]
        elif mv[0] == "t":
            i += 1
        else:
            j += 1
    return pairs


def page_boxes(page, lines: list[str]):
    """Boxes for the transcribed lines of one scan: per line None, or the boxes of the printed lines
    it holds as one flat list x0, y0, x1, y1, ... in ten-thousandths of the page; and the pairs."""
    W, H = page.rect.width, page.rect.height
    ocr = ocr_lines(page)
    pairs = pair([norm(l) for l in lines], [norm(o[1]) for o in ocr])
    out = [None] * len(lines)
    for i, j, k, s in pairs:
        if s >= MIN_SCORE and len(norm(lines[i])) >= 3:
            out[i] = [round(v * 10000 / d) for o in ocr[j:j + k] for v, d in zip(o[2], (W, H, W, H))]
    return out, pairs


VERSION = 3   # raise when the pairing changes, so cached boxes are made again


def boxes(pdf: Path, root):
    """{scan: [boxes or None per main-text line] (page_boxes)} for every page of book.xml that has text."""
    import pymupdf
    doc = pymupdf.open(str(pdf))
    res = {}
    for pg in root.iter("page"):
        n = int(pg.get("n"))
        lines = ["".join(l.itertext()) for l in pg.iterfind("main_text/l")]
        if not lines or n > doc.page_count:
            continue
        res[n], _ = page_boxes(doc[n - 1], lines)
    return res


def cached_boxes(book_dir: Path, root):
    """boxes() for a book, kept in <book_dir>/pdf_boxes.json until book.xml or the PDF changes
    (pairing a whole book takes minutes). None if the book has no local PDF."""
    pdf = pdf_of(book_dir, root)
    if pdf is None:
        return None
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        print("  pymupdf is not installed: no line boxes on the scans (pip install pymupdf)")
        return None
    key, text_pdf = cache_key(book_dir, pdf)
    cache = book_dir / "pdf_boxes.json"
    if cache.is_file():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("key") == key:
            return {int(k): v for k, v in d["boxes"].items()}
    print(f"  pairing the lines with the PDF's text layer ({text_pdf.name}{', OCR layer' if text_pdf != pdf else ''}) ...")
    res = boxes(text_pdf, root)
    cache.write_text(json.dumps({"key": key, "boxes": res}, separators=(",", ":")), encoding="utf-8")
    return res


def cache_key(book_dir: Path, pdf: Path, xml_sha: str | None = None) -> tuple[str, Path]:
    """The key of pdf_boxes.json for book.xml as it is (or with sha1 xml_sha), and the PDF with the text."""
    import hashlib
    sha = xml_sha or hashlib.sha1((book_dir / "book.xml").read_bytes()).hexdigest()
    key = f"{VERSION}-{sha}-{pdf.stat().st_size}"
    # A scan without a text layer of its own: ocr_tekstlaag.py (PRDL/scripts) saved one with the same
    # page sizes in <book_dir>/ocr_layer/<pdf>; the lines are paired with that text.
    text_pdf, from_layer = layer_or_pdf(book_dir, pdf)
    if from_layer:
        key += "-ocr_layer"
    return key, text_pdf


def patch_boxes(book_dir: Path, root, was_sha: str, scans) -> bool:
    """pdf_boxes.json after a change to book.xml on these scans only: they are paired again, the rest
    of the file stays, and it counts for the book.xml of now. was_sha: the sha1 of book.xml before the
    change. False, with the file left alone, when that was not the file's own book.xml: cached_boxes
    pairs the whole book then."""
    import pymupdf
    cache, pdf = book_dir / "pdf_boxes.json", pdf_of(book_dir, root)
    if pdf is None or not cache.is_file():
        return False
    d = json.loads(cache.read_text(encoding="utf-8"))
    if d.get("key") != cache_key(book_dir, pdf, was_sha)[0]:
        return False
    key, text_pdf = cache_key(book_dir, pdf)
    doc = pymupdf.open(str(text_pdf))
    pages = {int(pg.get("n")): pg for pg in root.iter("page")}
    for n in scans:
        lines = ["".join(l.itertext()) for l in pages[n].iterfind("main_text/l")]
        if lines and n <= doc.page_count:
            d["boxes"][str(n)], _ = page_boxes(doc[n - 1], lines)
    cache.write_text(json.dumps({"key": key, "boxes": d["boxes"]}, separators=(",", ":")), encoding="utf-8")
    return True


def pdf_of(book_dir: Path, root) -> Path | None:
    p = HERE.parents[1] / "frontend" / "pdfs" / (root.get("source") or "")
    return p if root.get("source") and p.is_file() else None


def main():
    from lxml import etree
    import pymupdf
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book_dir", type=Path)
    ap.add_argument("--pages", help="only these scans, e.g. 190-210")
    ap.add_argument("--show", type=int, help="print the pairing of one scan")
    args = ap.parse_args()
    root = etree.parse(str(args.book_dir / "book.xml"), etree.XMLParser(recover=True)).getroot()
    pdf = pdf_of(args.book_dir, root) or sys.exit("no local PDF for this book")
    doc = pymupdf.open(str(pdf))
    lo, hi = map(int, args.pages.split("-")) if args.pages else (1, 10 ** 6)
    tot = got = pages = 0
    worst = []
    for pg in root.iter("page"):
        n = int(pg.get("n"))
        lines = ["".join(l.itertext()) for l in pg.iterfind("main_text/l")]
        if not lines or not lo <= n <= hi or n > doc.page_count:
            continue
        bx, pairs = page_boxes(doc[n - 1], lines)
        k = sum(b is not None for b in bx)
        tot, got, pages = tot + len(lines), got + k, pages + 1
        worst.append((k / len(lines), n, k, len(lines)))
        if n == args.show:
            ocr = ocr_lines(doc[n - 1])
            print(f"scan {n}: {len(ocr)} OCR lines, {len(lines)} transcribed")
            at = {i: (j, k, s) for i, j, k, s in pairs}
            for i, l in enumerate(lines):
                j, k, s = at.get(i, (None, 0, 0))
                print(f"{'OK' if bx[i] else '--'} {s:.2f} x{k}  {l[:45]:45}  | {ocr[j][0][:45] if j is not None else ''}")
    worst.sort()
    print(f"{pages} scans, {tot} lines, {got} with a box ({100 * got / max(tot, 1):.1f}%)")
    full = sum(w[0] == 1 for w in worst)
    print(f"scans with every line: {full}; below 80%: {sum(w[0] < .8 for w in worst)}; below 50%: {sum(w[0] < .5 for w in worst)}")
    print("weakest scans:", ", ".join(f"{n} ({k}/{t})" for _, n, k, t in worst[:12]))


if __name__ == "__main__":
    main()
