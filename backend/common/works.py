"""A work, as the scripts name it and find it (REFACTOR_PLAN.md step 53).

A work has an id, W06290. A further volume of it is a work of its own for the site, with PRDL's print id:
W07610-02. Its folder is named after its PDF, W06290_Rivet_Commentarii_in_librum_secundum_Mosis_1634/, in data/ and
beside the typeset title pages alike. It is read from its transcription, book.xml, and when it has none from the
PDF's own OCR text, ocr_layer/book.xml.

The places that cut the volume off an id (work[:6]) do so on purpose or not; they keep doing it (parking list P25).
"""
import re
import sys
from pathlib import Path

from common.paths import DATA

WORK_ID = re.compile(r"W\d{5}(?:-\d+)?")   # W06290, or a further volume W07610-02


def outside_library(eid: str) -> bool:
    """Whether a work or edition lies outside the library of 1550-1750: the Opera Calvini (W99001) and the reference
    works (W99002 and on, reference_works/catalog.py). The statistics, the dashboard and the register leave them out."""
    return eid.startswith("W99")


def id_of(work, whole: bool = False) -> str:
    """A work as a script is given it, as its id (REFACTOR_PLAN.md step 97): a number (W06290, a further volume
    W07610-02), a folder name (W06290_Rivet_…, also a PDF's name) or a path (to the folder, or to a file in it, such
    as its book.xml or ocr_layer/). With whole=True without the volume (W07610), for the scripts that cut it off
    (work[:6], parking list P25). What names no work comes back as it was, so a filter then still finds nothing."""
    p = Path(str(work).rstrip("/"))
    for part in (p, *p.parents):
        m = WORK_ID.match(part.name)
        if m and part.name[m.end():m.end() + 1] in ("", "_", "."):
            return m[0][:6] if whole else m[0]
    return str(work)


def ids_of(works, whole: bool = False) -> set[str]:
    """id_of for each of a list of works (None or empty: an empty set)."""
    return {id_of(w, whole) for w in works or ()}


def text_of(folder: Path) -> Path:
    """The book.xml a work is read from: its transcription, else its OCR text. The second also when it is not
    there either: ask is_file()."""
    return folder / "book.xml" if (folder / "book.xml").is_file() else folder / "ocr_layer" / "book.xml"


def edition_of(book: Path) -> str:
    """The edition of a book: W06290 for data/W06290_…/ and data/W06290_…/ocr_layer/; a further volume of a work
    (data/W07610-02_…) is an edition of its own, W07610-02."""
    return (book.parent if book.name == "ocr_layer" else book).name.split("_")[0]


def folder_of(work: str) -> Path:
    """The folder of a work in data/ (DATA): W06290 or its whole folder name; the first if there are more. Stops the
    script when there is none. Was in toc_haiku.py and index_toc.py, the same twice (REFACTOR_PLAN.md step 53)."""
    found = sorted(DATA.glob(f"{work}_*")) or sorted(DATA.glob(work))
    if not found:
        sys.exit(f"no folder for {work} under {DATA}")
    return found[0]


def folder_in(top: Path, work: str) -> Path | None:
    """The first folder in top named after the work (W06290_…), or None."""
    return next((p for p in sorted(top.glob(f"{work}_*")) if p.is_dir()), None)


# --- which PDF has the text: a scan without a text layer of its own got one from ocr_tekstlaag.py (PRDL/scripts), a
# text-only PDF with the same page sizes beside the work's OCR text. Five places look for it, with three rules; each
# rule has its name here and its places keep it.

def layer_pdf(folder: Path, pdf_name: str) -> Path:
    """Where the text layer of a work's PDF is: <work folder>/ocr_layer/<the PDF's name>."""
    return folder / "ocr_layer" / pdf_name


def layer_if_same(layer: Path, pages: int):
    """The text layer, opened, when it is there and has as many pages as the scan; else None. For transcribe.py and
    layout_check.py, which then take a page from it where the scan has no text."""
    import pymupdf
    doc = pymupdf.open(layer) if layer.is_file() else None
    return doc if doc is not None and doc.page_count == pages else None


def layer_if_there(layer: Path):
    """The text layer, opened, when it is there, whatever its pages; else None. For profile.py (which asks for the
    same number of pages page by page) and figures.py (which does not)."""
    import pymupdf
    return pymupdf.open(layer) if layer.is_file() else None


def text_pdf(book_dir: Path, pdf: Path) -> tuple[Path, bool]:
    """The PDF a book's lines are paired with on the scans (enrich/pdfboxes.py): for the whole book its text layer
    when there is one, else the PDF itself; and whether it is the layer."""
    layer = layer_pdf(book_dir, pdf.name)
    return (layer, True) if layer.is_file() else (pdf, False)
