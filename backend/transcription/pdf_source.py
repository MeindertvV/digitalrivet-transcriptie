"""The pages of a book to transcribe: a PDF, with its OCR text layer, or a folder of images (REFACTOR_PLAN.md
step 72: was the class Book of transcribe.py, which still has the name).

Book gives each page as an image to send (whole, or a part of it: parts.py), its printed lines from the text layer
(split_lines.py) and all of that layer's text. A scan without a text layer of
its own takes the separate one in data/<work>/ocr_layer/ when it has as many pages.
"""

import re
import sys
import threading
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.paths import ocr_layer  # noqa: E402
from common.works import layer_if_same  # noqa: E402
from transcription import split_lines, parts  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


class Book:
    """Uniform access to the pages of a PDF or a folder of images."""

    def __init__(self, source: Path):
        self.source = source
        self.lock = threading.Lock()  # PyMuPDF is not thread-safe
        if source.is_dir():
            self.files = sorted(
                (p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                key=lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)],
            )
            self.doc = None
            self.count = len(self.files)
        else:
            self.doc = pymupdf.open(source)
            self.count = self.doc.page_count
            # A scan without a text layer of its own: ocr_tekstlaag.py (PRDL/scripts) saved one as a
            # text-only PDF with the same page sizes in data/<work>/ocr_layer/<work>.pdf.
            self.layer = layer_if_same(ocr_layer(source.stem) / source.name, self.count)
        if not self.count:
            sys.exit(f"No pages found in {source}")
        self.ocr_cache: dict[int, list[str] | None] = {}
        self.part_cache: dict[int, list[parts.Part]] = {}

    def text_page(self, n: int):
        """Page n of the PDF, or of its separate OCR layer when the scan itself has no text, or
        only text that split_lines rejects as noise (a scanner's OCR that failed on that page)."""
        page = self.doc[n - 1]
        if self.layer is not None and (len(page.get_text().strip()) < 20
                                       or split_lines.printed_lines(page) is None):
            return self.layer[n - 1]
        return page

    def ocr(self, n: int) -> list[str] | None:
        """The printed lines of page n from the PDF's text layer (split_lines.printed_lines), or
        None for images and pages without a usable text layer."""
        if self.doc is None:
            return None
        with self.lock:
            if n not in self.ocr_cache:
                self.ocr_cache[n] = split_lines.printed_lines(self.text_page(n))
            return self.ocr_cache[n]

    def render_jpeg(self, n: int, dpi: int, max_px: int) -> bytes:
        """Render page n (1-based) so its long edge is at most max_px pixels."""
        def render(page) -> bytes:
            long_side = max(page.rect.width, page.rect.height)
            zoom = min(dpi / 72, max_px / long_side)
            return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("jpeg", jpg_quality=90)

        with self.lock:
            if self.doc is not None:
                return render(self.doc[n - 1])
            with pymupdf.open(self.files[n - 1]) as img:  # an image file: close it again
                return render(img[0])

    def parts(self, n: int) -> list[parts.Part]:
        """How to read page n: its parts (parts.py), or [] to send it whole."""
        if self.doc is None:
            return []
        with self.lock:
            if n not in self.part_cache:
                self.part_cache[n] = parts.plan(self.text_page(n))
            return self.part_cache[n]

    def ocr_text(self, n: int) -> str | None:
        """All the text of page n's text layer, margins included."""
        if self.doc is None:
            return None
        with self.lock:
            return self.text_page(n).get_text()

    def render_part(self, n: int, part: "parts.Part", dpi: int, max_px: int) -> bytes:
        """Render a part of page n so its long edge is at most max_px pixels."""
        with self.lock:
            clip = pymupdf.Rect(part.clip)
            zoom = min(dpi / 72, max_px / max(clip.width, clip.height))
            pix = self.doc[n - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
            for x0, y0, x1, y1 in part.blank:   # a clipped pixmap keeps the page's coordinates
                box = pymupdf.IRect(int(x0 * zoom), int(y0 * zoom), int(x1 * zoom) + 1, int(y1 * zoom) + 1) & pix.irect
                pix.set_rect(box, (255, 255, 255))
            return pix.tobytes("jpeg", jpg_quality=90)
