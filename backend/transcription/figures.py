"""Pictures on a page cut from the scan as PNGs with a transparent background, for the reader.

The model writes a picture as <figure kind=".." place="..">description</figure> (prompts/layout/
_common.txt, rule B); ocr_split.py keeps it as <l type="figure" kind=".." place=".." alt="..">
in the page's lines, unnumbered. Where the picture is, the transcription does not say. Looking for
it in the ink of the scan without the OCR's words does not work: much printed text has no OCR word
and passes for a picture. So locate() asks Haiku once more, with the page at 1000 px and the
descriptions, for a box around each drawing (about $0.0003 a page; in a trial it found the
diagrams well but cut them a little tight), and cut() widens it by PAD. Each picture becomes
data/<work>/figures/pNNNN_K.png: dark ink whose opacity is how much darker the scan is than the
paper there, so paper and show-through are gone and the picture sits on any background (in dark
mode the reader inverts it). The <l> gets src="figures/pNNNN_K.png" and box="x0 y0 x1 y1"
(share of the page).

  python transcription/figures.py <work> [--pages 12,40]     # again for the saved pages
"""

import argparse
import base64
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.pagespec import page_numbers  # noqa: E402
from common.paths import DATA, PDF_DIR, ocr_layer, refuse_foreign  # noqa: E402
from common.works import folder_in, id_of, layer_if_there  # noqa: E402
from transcription import split_lines  # noqa: E402  (is_decoration)

ASK_PX = 1000         # long edge of the page Haiku sees
PAD = 0.025           # share of the page added around Haiku's box: it cuts a little tight
CUT_DPI = 200
PAPER = 40            # grey levels below the paper where ink starts: show-through stays out
CLIENT = None
SPENT = 0.0           # $ spent on locate() in this process
VIA = "api"           # "claude": ask through the subscription (transcribe.py --via claude sets this)


def locate(scan, alts: list[str]) -> list[tuple[float, float, float, float] | None]:
    """A box (share of the page, x0 y0 x1 y1) per picture, in the order of alts, from Haiku."""
    global CLIENT, SPENT
    from transcription import layout_profile as profile   # the client and the one model
    zoom = ASK_PX / max(scan.rect.width, scan.rect.height)
    pix = scan.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    listing = "; ".join(f"{i}. {a or 'a picture'}" for i, a in enumerate(alts, 1))
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.standard_b64encode(pix.tobytes("jpeg", jpg_quality=85)).decode()}},
        {"type": "text", "text": f"This page image is {pix.width}x{pix.height} pixels. It holds these pictures: {listing}. "
                                 "For each, in that order, give the box around the drawing itself, not the text beside "
                                 "or below it, in pixels. JSON only: [[x0,y0,x1,y1], ...]"}]
    if VIA == "claude":
        from common import via_claude
        msg = via_claude.ask("You locate pictures on scanned book pages.", content, profile.MODEL)
    else:
        CLIENT = CLIENT or profile.client_()
        msg = CLIENT.messages.create(model=profile.MODEL, max_tokens=1500, messages=[{"role": "user", "content": content}])
        SPENT += (msg.usage.input_tokens * profile.PRICE[0] + msg.usage.output_tokens * profile.PRICE[1]) / 1e6
    text = "".join(b.text for b in msg.content if b.type == "text")
    try:
        boxes = json.loads(text[text.index("["):text.rindex("]") + 1])
    except ValueError:
        return [None] * len(alts)
    out = []
    for b in boxes[:len(alts)]:
        try:
            x0, y0, x1, y1 = (float(v) for v in b)
        except (TypeError, ValueError):
            out.append(None)
            continue
        box = (x0 / pix.width, y0 / pix.height, x1 / pix.width, y1 / pix.height)
        ok = 0 <= box[0] < box[2] <= 1.02 and 0 <= box[1] < box[3] <= 1.02 and (box[2] - box[0]) * (box[3] - box[1]) > 0.003
        out.append(box if ok else None)
    return out + [None] * (len(alts) - len(out))


def transparent(scan, box) -> Image.Image:
    """The scan inside box as black ink on nothing: opacity = darkness against the paper."""
    r = scan.rect
    m = PAD
    clip = pymupdf.Rect((box[0] - m) * r.width, (box[1] - m) * r.height,
                        (box[2] + m) * r.width, (box[3] + m) * r.height) & r
    zoom = CUT_DPI / 72
    pix = scan.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, colorspace=pymupdf.csGRAY, alpha=False)
    g = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(np.float32)
    paper = np.percentile(g, 90)
    dark = min(np.percentile(g, 1), paper - PAPER - 30)
    a = np.clip((paper - PAPER - g) / max(paper - PAPER - dark, 1), 0, 1)
    alpha = (np.power(a, 0.8) * 255).astype(np.uint8)
    return Image.merge("LA", (Image.new("L", (pix.width, pix.height), 20), Image.fromarray(alpha)))


def cut(scan, text_page, xml: str, n: int, out_dir: Path) -> str:
    """The page's figure lines with src and box set, their PNGs written to out_dir/figures/.
    text_page is not used any more (see the module doc), kept so callers need not change."""
    if 'type="figure"' not in xml:
        return xml
    page = ET.fromstring(xml)
    for parent in list(page.iter()):   # a woodcut or decorated initial is no picture (split_lines.is_decoration)
        for l in [l for l in parent if l.tag == "l" and l.get("type") == "figure"
                  and split_lines.is_decoration(l.get("kind"), l.get("alt") or "")]:
            parent.remove(l)
    figs = [l for l in page.iter("l") if l.get("type") == "figure"]
    for old in (out_dir / "figures").glob(f"p{n:04d}_*.png"):
        old.unlink()
    if not figs:
        if (out_dir / "figures").is_dir() and not any((out_dir / "figures").iterdir()):
            (out_dir / "figures").rmdir()   # an empty folder would count as pictures cut (ops/pijplijn/stand.py)
        return ET.tostring(page, encoding="unicode")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    try:
        boxes = locate(scan, [l.get("alt") or l.get("kind") or "" for l in figs])
    except Exception as e:   # no picture is better than no page
        print(f"  page {n}: pictures not located: {e}")
        return xml
    for k, (l, box) in enumerate(zip(figs, boxes), 1):
        if box is None:
            continue
        name = f"figures/p{n:04d}_{k}.png"
        transparent(scan, box).save(out_dir / name, optimize=True)
        l.set("src", name)
        l.set("box", " ".join(f"{v:.3f}" for v in box))
    return ET.tostring(page, encoding="unicode")


def wanted(spec: str | None) -> set[int]:
    """The scans --pages names ("20-24,30"); none named is an empty set: then every scan with a figure."""
    return set(page_numbers(spec)) if spec else set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--pages")
    args = ap.parse_args()
    args.work = (folder_in(DATA, id_of(args.work)) or Path(args.work)).name
    root = DATA / args.work
    refuse_foreign(root)
    doc = pymupdf.open(PDF_DIR / f"{args.work}.pdf")
    layer = layer_if_there(ocr_layer(args.work) / f"{args.work}.pdf")
    want = wanted(args.pages)
    for f in sorted((root / "pages").glob("p????.xml")):
        n = int(f.stem[1:])
        if want and n not in want:
            continue
        xml = f.read_text(encoding="utf-8")
        if 'type="figure"' not in xml:
            continue
        scan = doc[n - 1]
        tp = layer[n - 1] if layer is not None and len(scan.get_text().strip()) < 20 else scan
        new = cut(scan, tp, xml, n, root)
        f.write_text(new + "\n", encoding="utf-8")
        print(f"page {n}: {new.count('src=')} picture(s) cut")
    print(f"Spent on locating: ${SPENT:.4f}")


if __name__ == "__main__":
    main()
