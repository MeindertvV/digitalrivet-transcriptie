"""Compare the transcription with the PDF's own text layer (the scanner's OCR), line by line.

Every transcribed main-text line of book.xml is paired with the OCR line (or lines) it stands for
on the scan, with the same pairing the site uses to mark a line on the scan (enrich/pdfboxes.py),
so the lines here are exactly the lines of the reader. Per line the score is the share of letters
that agree once long s/f, u/v, i/j and ligatures are made equal. The OCR is poor, so a line rarely
scores 100%; what matters is a line that scores much lower than the lines around it.

  goed        score >= 0.85
  redelijk    score >= 0.55   (the threshold at which the site marks a line on the scan)
  afwijkend   score <  0.55   paired, but the letters hardly agree
  geen        no OCR line found for it (skipped by the OCR, or a line the transcription invented)
  (no score)  the scan has no usable text layer

Writes in the book folder
  ocr_regels.json   per scan per line [score 0-100, OCR text] or null; build_reader.py puts it on
                    the site (switch "OCR-controle" in the reader)
  ocr_regels.html   the report: every line beside its OCR, differing words marked

A book with ocr_regels.json counts as checked against the OCR layer; it stays in data/<book>/.

Examples
  python ocr_check/line_compare.py data/W01206_Calvin_Institutio_Christianae_religionis_in_libros_quatuor_1561
  python ocr_check/line_compare.py data/W06290_Rivet_… --pages 20-60
"""

import argparse
import hashlib
import html
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

from lxml import etree
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.pagespec import page_numbers  # noqa: E402
from common.paths import PDF_DIR  # noqa: E402
from common import lineboxes  # noqa: E402  (ocr_lines, pair, norm: the site's pairing)

GOOD, FAIR = 0.85, lineboxes.MIN_SCORE
MIN_OCR_WORDS = 15   # fewer words with 3+ letters on a scan: no usable text layer
MIN_MISSING = 15     # an OCR line left over counts as missing text from this many letters
WEAK_PAGE = 0.80     # paired lines scoring less than this on average: the OCR does not follow the page
                     # (an index in columns), so its left-over lines are not reported
VERSION = 1


def grade(s) -> str:
    if s is None:
        return "geen"
    return "goed" if s >= GOOD else "redelijk" if s >= FAIR else "afwijkend"


def furniture(pg) -> list[str]:
    """Normalised text of everything on a scan that is not main text: running head, page number,
    signature, catchword, margin notes. OCR lines like these are not missing text."""
    out = []
    for el in pg:
        if el.tag != "main_text":
            out += [lineboxes.norm(t) for t in el.itertext()]
    for el in pg.iterfind("main_text//m"):   # margin notes inside a line
        out.append(lineboxes.norm("".join(el.itertext())))
    return [t for t in out if len(t) >= 3]


def garbage(t: str) -> bool:
    """OCR noise rather than a line of text: too few of its characters are letters, or too few of
    its words look like words (3+ letters, no digits or stray signs inside)."""
    chars = [c for c in t if not c.isspace()]
    words = t.split()
    good = [w for w in words if len(w.strip(".,;:?!()'\"-")) >= 3 and w.strip(".,;:?!()'\"-").isalpha()]
    return (sum(c.isalpha() for c in chars) < 0.85 * len(chars)) or len(good) < 0.5 * len(words)


def elsewhere(s: str, other: str) -> bool:
    """The letters of an OCR line are in the transcription of the page after all, at another place
    (lines merged, columns read in another order): not missing. Counts only stretches of 5+ letters
    in common, per piece of 40 letters, as check_ocr.py does."""
    found = 0
    for chunk in (s[i:i + 40] for i in range(0, len(s), 40)):
        sm = SequenceMatcher(None, chunk, other, autojunk=False)
        found += sum(m.size for m in sm.get_matching_blocks() if m.size >= 5)
    return found >= 0.7 * len(s)


def page_lines(pdf_page, lines: list[str], extra: list[str]):
    """{"r": per transcribed line [score, OCR text], [0, ""] where no OCR line was paired with it,
    or None for a line without letters; "x": [[i, OCR text]] for OCR lines that no transcribed line
    stands for, after transcribed line i (-1: before the first); "weak": true, and no "x", where the
    OCR does not follow the page (WEAK_PAGE)}. None when the scan has no usable
    text layer."""
    ocr = lineboxes.ocr_lines(pdf_page)
    if sum(len(lineboxes.norm(w)) >= 3 for o in ocr for w in o[1].split()) < MIN_OCR_WORDS:
        return None
    on = [lineboxes.norm(o[1]) for o in ocr]
    pairs = lineboxes.pair([lineboxes.norm(l) for l in lines], on)
    rows = [[0, ""] if len(lineboxes.norm(l)) >= 3 else None for l in lines]
    after, used = [-1] * len(ocr), set()   # after: the transcribed line paired before each OCR line
    for i, j, k, s in pairs:
        if rows[i] is not None:
            rows[i] = [round(s * 100), " ".join(o[1] for o in ocr[j:j + k])]
        used.update(range(j, j + k))
        for jj in range(j, len(ocr)):
            after[jj] = i
    sc = [r[0] for r in rows if r and r[1]]
    if sc and sum(sc) / len(sc) < WEAK_PAGE * 100:
        return {"r": rows, "x": [], "weak": True}
    page = "".join(lineboxes.norm(l) for l in lines) + "".join(extra)
    x = [[after[j], ocr[j][1]] for j in range(len(ocr))
         if j not in used and len(on[j]) >= MIN_MISSING and not garbage(ocr[j][1])
         and not any(lineboxes.similarity(on[j], e) >= 0.6 or on[j] in e for e in extra)
         and not elsewhere(on[j], page)]
    return {"r": rows, "x": x}


def word_diff(trans: str, ocr: str) -> tuple[str, str]:
    """Both texts as html, the words the other side does not have in <mark>."""
    a, b = trans.split(), ocr.split()
    sm = SequenceMatcher(None, [lineboxes.norm(w) for w in a], [lineboxes.norm(w) for w in b], autojunk=False)
    ha, hb = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        wa, wb = html.escape(" ".join(a[i1:i2])), html.escape(" ".join(b[j1:j2]))
        if op == "equal":
            ha.append(wa); hb.append(wb)
        else:
            if wa: ha.append(f"<mark>{wa}</mark>")
            if wb: hb.append(f"<mark>{wb}</mark>")
    return " ".join(ha), " ".join(hb)


REPORT_CSS = """
:root{--bg:#fffdf8;--ink:#2a241c;--muted:#857a69;--rule:#e4dccc;--goed:#2f7d32;--redelijk:#a07000;
--afwijkend:#b3261e;--geen:#7a5aa8;--mark:#f7d77a}
@media (prefers-color-scheme:dark){:root{--bg:#1f1b15;--ink:#e9e1d2;--muted:#9a8f7d;--rule:#383126;
--goed:#7cc47f;--redelijk:#e0b54a;--afwijkend:#f08a80;--geen:#b9a0e0;--mark:#6b5414}}
body{font:15px/1.45 Georgia,serif;max-width:1300px;margin:2em auto;padding:0 16px;color:var(--ink);background:var(--bg)}
h1{font-size:1.4em} h3{margin:1.6em 0 .3em;font-size:1.05em} a{color:inherit}
table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid var(--rule);padding:3px 6px;vertical-align:top;text-align:left}
th{font:600 12px system-ui;color:var(--muted)} td.n,td.s{font:12px system-ui;white-space:nowrap;color:var(--muted)}
td.s{text-align:right;font-weight:600} td.o{font-family:ui-monospace,monospace;font-size:13px;color:var(--muted)}
tr.goed td.s{color:var(--goed)} tr.redelijk td.s{color:var(--redelijk)} tr.afwijkend td.s{color:var(--afwijkend)}
tr.geen td.s{color:var(--geen)} tr.alleen td.s{color:var(--afwijkend)} td.gap{color:var(--muted);font-style:italic}
tr.afwijkend td:first-child,tr.geen td:first-child,tr.alleen td:first-child{box-shadow:inset 3px 0 var(--afwijkend)}
mark{background:var(--mark);color:inherit;border-radius:2px} .sum span{margin-right:1.4em;font:13px system-ui}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:.5em 0 1em;max-width:600px}
.bar i{display:block} p.no{color:var(--muted);font-style:italic}
label{font:13px system-ui;color:var(--muted)} body.weak tr.goed,body.weak tr.redelijk{display:none}
body.weak section:not(:has(tr.afwijkend,tr.geen,tr.alleen)){display:none}
"""


def write_report(path: Path, pdf: Path, res: dict, numbers: dict, texts: dict, summary: dict):
    c = summary["counts"]
    tot = sum(c.values()) or 1
    colors = {"goed": "var(--goed)", "redelijk": "var(--redelijk)", "afwijkend": "var(--afwijkend)", "geen": "var(--geen)"}
    out = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>OCR per regel</title><style>{REPORT_CSS}</style>",
           f"<h1>OCR per regel: {html.escape(pdf.stem)}</h1>",
           f"<p>{summary['scans']} scans met tekst, {summary['scans_ocr']} met bruikbare OCR, waarvan "
           f"{summary['scans_weak']} waar de OCR de pagina slecht volgt. "
           f"{summary['lines']} regels vergeleken, gemiddelde overeenkomst {summary['mean']}%.</p>",
           "<div class=bar>" + "".join(f"<i style='width:{100 * c[k] / tot}%;background:{colors[k]}' title='{k}: {c[k]}'></i>"
                                       for k in colors) + "</div>",
           "<p class=sum>" + "".join(f"<span style='color:{colors[k]}'>{k}: {c[k]}</span>" for k in colors)
           + f"<span>OCR-regels zonder transcriptie: {summary['ocr_only']}</span></p>",
           "<label><input type=checkbox onchange=\"document.body.classList.toggle('weak',this.checked)\"> "
           "alleen afwijkende regels</label>"]
    uri = pdf.resolve().as_uri()
    for n in sorted(texts):
        p = res.get(n)
        num = numbers.get(n, "")
        out.append(f"<section><h3><a href='{uri}#page={n}' target=_blank>scan {n}</a>"
                   + (f" · p. {html.escape(num)}" if num else "") + "</h3>")
        if p is None:
            out.append("<p class=no>geen bruikbare OCR op deze scan</p></section>")
            continue
        if p.get("weak"):
            out.append("<p class=no>de OCR volgt deze scan slecht (kolommen?): regels die alleen in de OCR staan "
                       "worden hier niet gemeld</p>")
        out.append("<table><tr><th>regel</th><th>transcriptie</th><th>OCR</th><th>overeenkomst</th></tr>")
        miss = {}
        for i, o in p["x"]:
            miss.setdefault(i, []).append(o)

        def missing(i):
            for o in miss.get(i, []):
                out.append(f"<tr class=alleen><td class=n></td><td class=gap>ontbreekt in de transcriptie?</td>"
                           f"<td class=o><mark>{html.escape(o)}</mark></td><td class=s>alleen OCR</td></tr>")

        missing(-1)
        for i, ((ln, t), r) in enumerate(zip(texts[n], p["r"])):
            if r is None:
                out.append(f"<tr><td class=n>{ln}</td><td>{html.escape(t)}</td><td class=o></td><td class=s></td></tr>")
            else:
                s, o = r
                g = grade(s / 100 if o else None)
                ht, ho = word_diff(t, o) if o else (html.escape(t), "")
                out.append(f"<tr class={g}><td class=n>{ln}</td><td>{ht}</td><td class=o>{ho}</td>"
                           f"<td class=s>{f'{s}%' if o else 'geen'}</td></tr>")
            missing(i)
        out.append("</table></section>")
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book", type=Path, help="map met book.xml")
    ap.add_argument("--pdf", type=Path, help=f"standaard: {PDF_DIR}/<source uit book.xml>")
    ap.add_argument("--pages", help="alleen deze scans, bv. 20-60,75")
    ap.add_argument("--no-move", action="store_true", help=argparse.SUPPRESS)  # old option: books are no longer moved
    args = ap.parse_args()

    book = args.book.resolve()
    root = etree.parse(str(book / "book.xml"), etree.XMLParser(recover=True)).getroot()
    pdf = args.pdf or PDF_DIR / (root.get("source") or book.name + ".pdf")
    if not pdf.exists():
        sys.exit(f"PDF niet gevonden: {pdf} (geef --pdf)")
    doc = pymupdf.open(str(pdf))

    want = None
    if args.pages:
        want = set(page_numbers(args.pages))

    res, texts, numbers = {}, {}, {}
    pages = [pg for pg in root.iter("page") if pg.find("main_text/l") is not None
             and (want is None or int(pg.get("n")) in want)]
    for k, pg in enumerate(pages, 1):
        n = int(pg.get("n"))
        ls = list(pg.iterfind("main_text/l"))
        lines = ["".join(l.itertext()) for l in ls]
        texts[n] = [(l.get("n") or i + 1, " ".join(t.split())) for i, (l, t) in enumerate(zip(ls, lines))]
        numbers[n] = (pg.findtext("number") or "").strip()
        res[n] = page_lines(doc[n - 1], lines, furniture(pg)) if n <= doc.page_count else None
        if k % 50 == 0:
            print(f"  {k}/{len(pages)} scans", file=sys.stderr)

    counts = {"goed": 0, "redelijk": 0, "afwijkend": 0, "geen": 0}
    scores = []
    for p in res.values():
        for r in (p or {}).get("r", []):
            if r is None:
                continue
            counts[grade(r[0] / 100 if r[1] else None)] += 1
            scores.append(r[0])
    summary = {"scans": len(res), "scans_ocr": sum(r is not None for r in res.values()),
               "scans_weak": sum(bool(r and r.get("weak")) for r in res.values()),
               "lines": len(scores), "ocr_only": sum(len(p["x"]) for p in res.values() if p), "mean": round(sum(scores) / len(scores)) if scores else 0, "counts": counts}

    (book / "ocr_regels.json").write_text(json.dumps(
        {"version": VERSION, "book_xml": hashlib.sha1((book / "book.xml").read_bytes()).hexdigest(), "pdf": pdf.name, "good": round(GOOD * 100), "fair": round(FAIR * 100),
         "summary": summary, "pages": res}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    write_report(book / "ocr_regels.html", pdf, res, numbers, texts, summary)
    doc.close()

    print(f"{summary['scans_ocr']}/{summary['scans']} scans met OCR ({summary['scans_weak']} slecht gevolgd), {summary['lines']} regels, "
          f"gemiddeld {summary['mean']}%")
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f"; OCR-regels zonder transcriptie {summary['ocr_only']}")

    print(f"Rapport: {book / 'ocr_regels.html'}")


if __name__ == "__main__":
    main()
