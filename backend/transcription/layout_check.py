"""Does a scanned work have a plain layout (one text column, light marginalia)?

From the OCR word positions only, no API. On 12 sample pages between 20% and 80% of the book (from
the ocr_layer PDF where the scan has no text), the lines that span most of the text block make up
the main column. A page where those carry under 35% of the characters scores 1.0 (columns, a table,
a word list); otherwise its score is the share of characters in lines outside the main column.
Verdict: "columns" if the median page scores 0.5+, "marginalia" if a quarter of the pages score
0.15+, otherwise plain. budget_run.py skips works with a verdict.

  python transcription/layout_check.py W05794_... [W...]      # verdict and per-page values
"""

import statistics
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.paths import DATA, PDF_DIR  # noqa: E402
from common.works import layer_if_same, layer_pdf  # noqa: E402

SAMPLE = 12


def page_side(page) -> float | None:
    words = page.get_text("words")
    if len(words) < 80:
        return None   # title page, plate, blank: says nothing about the layout
    x0 = sorted(w[0] for w in words)[len(words) // 50]          # robust text-block edges
    x1 = sorted(w[2] for w in words)[-(len(words) // 50) - 1]
    lines: dict[tuple[int, int], list] = {}
    for w in words:
        lines.setdefault((w[5], w[6]), []).append(w)
    spans = [(min(w[0] for w in ws), max(w[2] for w in ws), sum(len(w[4]) for w in ws))
             for ws in lines.values()]
    total = sum(n for *_, n in spans)
    # The main column: the lines that span most of the text block.
    long = [(a, b, n) for a, b, n in spans if b - a >= 0.55 * (x1 - x0)]
    if sum(n for *_, n in long) < 0.35 * total:
        return 1.0   # hardly any full-width line: columns, a table, a word list
    m0 = statistics.median(a for a, _, _ in long)
    m1 = statistics.median(b for _, b, _ in long)
    tol = 0.03 * (m1 - m0)
    # Margin text: lines that lie for the most part outside the main column.
    margin = sum(n for a, b, n in spans if (a + b) / 2 < m0 - tol or (a + b) / 2 > m1 + tol)
    return margin / total if total else None


def score(work: str) -> tuple[float | None, list[float]]:
    doc = pymupdf.open(PDF_DIR / f"{work}.pdf")
    layer = layer_if_same(layer_pdf(DATA / work, f"{work}.pdf"), doc.page_count)
    n = doc.page_count
    picks = sorted({int(n * (0.2 + 0.6 * i / (SAMPLE - 1))) for i in range(SAMPLE)})
    vals = []
    for i in picks:
        page = doc[i]
        if layer is not None and len(page.get_text().strip()) < 20:
            page = layer[i]
        v = page_side(page)
        if v is not None:
            vals.append(v)
    return (statistics.median(vals) if len(vals) >= 3 else None), vals


def verdict(work: str) -> tuple[str | None, list[float]]:
    """'columns' (most pages without full-width lines), 'marginalia' (a quarter of the pages
    with 15%+ of their text beside the main column), or None for a plain layout / no OCR."""
    s, vals = score(work)
    if s is None:
        return None, vals
    if s >= 0.5:
        return "columns", vals
    if sum(0.15 <= v < 1 for v in vals) >= 0.25 * len(vals):
        return "marginalia", vals
    return None, vals


if __name__ == "__main__":
    for w in sys.argv[1:]:
        try:
            v, vals = verdict(w)
        except Exception as e:
            print(f"error\t{w}\t{e}")
            continue
        print(f"{v or 'plain'}\t{w}\t{' '.join(f'{x:.2f}' for x in vals)}")
