"""Where everything lives. Every script imports its folders from here, so a move is one edit.

data/<work>/            one folder per work (W06290_Rivet_…_1634), named after its PDF:
  book.xml book.json    the transcription (transcription/transcribe.py)
  ocr_layer/            the PDF's own OCR as book.xml, untranscribed (ocr/ocr_naar_xml.py)
data/_library/          the answers Haiku was paid for that belong to no single work (library(), below)
rag_index/ (next to data/)  the passages of the translations and their vectors, for the search on meaning (semantic_search/)
"""

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
PROMPTS = BACKEND / "prompts"
KEYS = BACKEND / "keys"
LOGS = BACKEND / "logs"
FRONTEND = BACKEND.parent / "frontend"
PILOT = BACKEND / "contents" / "reviewed"   # <work>/structuur.json: the reviewed contents lists (contents/structure.py reads them first)
if sys.platform == "win32":
    DATA = Path("E:/DigitalRivetSSD/backend/data")   # on the SSD; backend/data is a junction to it
    PDF_DIR = Path("E:/PRDL_pdfs")
    TESSDATA = Path("E:/tessdata")                   # lat.traineddata for PyMuPDF's Tesseract
else:                                                # the Linux server: /data/prdl/DigitalRivet
    DATA = BACKEND / "data"
    PDF_DIR = FRONTEND / "pdfs"                      # symlink to /data/prdl/pdfs
    TESSDATA = Path("/data/prdl/tessdata")
if os.environ.get("PRDL_DATA"):                       # elsewhere: a copy of some books, for tests and trial runs
    DATA = Path(os.environ["PRDL_DATA"])
RAG = DATA.resolve().parent / "rag_index"            # next to the data: the passages and their vectors (semantic_search/)
if os.environ.get("RAG_INDEX"):                      # elsewhere: the search service's container (semantic_search/serve.py)
    RAG = Path(os.environ["RAG_INDEX"])
# OCR layers made on the PC (ops/night/ocr_layers_pc.py) wait here until ops/night/ocr_layers.py takes them in
OCR_INBOX = Path(os.environ.get("PRDL_OCR_INBOX", "/data/prdl/ocr_inbox"))
OPERA = Path("/data/prdl/opera_calvini")              # the Opera Calvini's own tree, outside the books (opera_calvini/)
OPERA_PDFS = OPERA / "pdfs"                          # the tomes as scanned
OPERA_DATA = OPERA / "data"                          # what the steps of opera_calvini/ make of them
OPERA_TOMES = OPERA_DATA / "tomes"                   # extract.py: the text of each tome (t<NN>.json.gz)
OPERA_ALIGNED = OPERA_DATA / "al_blocks"             # align.py's out folder, with the cache of the tomes' words
OPERA_COMPARED = OPERA_DATA / "cmp_blocks"           # compare.py's out folder, as write_site.py read it
OPERA_CHECK = OPERA_DATA / "check"                   # what holds of the scan check: oc_fixes.json, verdicts.json, confusions.json
OPERA_CHOICE = OPERA_DATA / "check_choice"           # check_haiku.py run --choice: its items and results
REFERENCE_SRC = Path("/data/prdl/naslag/src")       # the reference works' scans as fetched (reference_works/fetch_pdfs.py)
BIBLE_SRC = Path("/data/prdl/bijbel/src")           # the Bible's texts as fetched (bible_text/fetch.py)
BIBLE_TEXT = FRONTEND / "data" / "bible"            # <book 0..65>/<chapter>.json: the verses in five versions (bible_text/build.py)
VERSIFICATION = BACKEND / "references" / "versification.json"   # the numberings, in the repo (bible_text/build.py --table)


def ocr_layer(work: str) -> Path:
    """data/<work>/ocr_layer/: book.xml, book.json, pdf_boxes.json from the PDF's OCR (ocr_naar_xml.py)."""
    return DATA / work / "ocr_layer"


def bible_books_of(folder: Path) -> Path:
    """data/<work>/bible_books.json: the Bible books a commentary volume treats (contents/bible_books.py); the
    catalogue (sitebuild/build_frontpage.py) takes them over Haiku's."""
    return folder / "bible_books.json"


def ocr_formatted(work: str) -> Path:
    """data/<work>/ocr_layer/formatted/: pages/ and raw/ cleaned by a text model (format_ocr.py)."""
    return ocr_layer(work) / "formatted"


ROOT = BACKEND.parent   # this checkout

# --- beside the code and outside git: what is kept between runs ---
# Each takes the checkout it belongs to, ROOT unless given. A build script in sitebuild/ passes the folder above
# it, as it always took its folders from where it stands: tests/golden/site.py runs copies of them in a small
# tree of their own, with a cache and title pages of its own.


def build_cache(root: Path = ROOT) -> Path:
    """.build_cache/: the workbook as read, word counts, page counts, the register per book, the duplicates scan.
    All of it can be made again without a model; what was paid for is in library()."""
    return root / ".build_cache"


def library() -> Path:
    """data/_library/: what a model was paid for and belongs to no single work, kept with the books so that the
    backup has it (REFACTOR_PLAN.md step 70; it was in .build_cache/). Its name is no work number, so no script takes
    it for a work."""
    return DATA / "_library"


def genre_answers() -> Path:
    """Subject and kind of every work as Haiku read them: subjects/genre_haiku.py writes it, the catalogue, the
    commentary contents and the duplicates scan read it."""
    return library() / "haiku_genres.json"


def loci_answers() -> Path:
    """The locus Haiku read at each heading: subjects/loci_haiku.py writes it, the register reads it."""
    return library() / "loci_haiku.json"


def titlepages(root: Path = ROOT) -> Path:
    """The typeset title pages, beside the checkout (/data/prdl/Titelpage maker/PRDL_titlepages): the reader's."""
    return root.parent / "Titelpage maker" / "PRDL_titlepages"


def titlepages_cli(root: Path = ROOT) -> Path:
    """What titlepages/make.py makes, beside the reader's (/data/prdl/Titelpage maker/PRDL_titlepages_cli), to look at
    before they go there."""
    return titlepages(root).with_name("PRDL_titlepages_cli")


# --- the gate against writing from another checkout (REFACTOR_PLAN.md P4) ---
# In the dev copy backend/data and frontend/data link to the live ones: a script run there would write in the live
# data with code that is not live. Every script that writes books asks first.

WRITE_LIVE = "PRDL_WRITE_LIVE"   # =1: write anyway


def checkout_of(path: Path) -> Path | None:
    """The checkout a path lies in once resolved (the first folder above it with backend/, PRDL/ and .git), or None."""
    path = Path(path).resolve()
    return next((p for p in (path, *path.parents)
                 if (p / "backend").is_dir() and (p / "PRDL").is_dir() and (p / ".git").exists()), None)


def refuse_foreign(*places: Path) -> None:
    """Stop the script if one of the places it is about to write in lies in another checkout than this code's: from
    the dev copy, the live data. A copy elsewhere (PRDL_DATA, a test's folder) lies in none and passes;
    PRDL_WRITE_LIVE=1 lets the run write anyway."""
    if os.environ.get(WRITE_LIVE) == "1":
        return
    for place in places:
        home = checkout_of(place)
        if home is not None and home != ROOT:
            sys.exit(f"{place} is the data of {home}, not of this checkout ({ROOT}): run it there, "
                     f"or on a copy (PRDL_DATA), or with {WRITE_LIVE}=1")


# --- the files of the duplicates (backend/docs/DUBBELEN.md), below a checkout ---
# The gate (common/dubbel.py) reads them in the checkout that owns the books (owner()), the scan (duplicates/scan.py)
# writes them in its own: two bases, kept as they were.


def owner() -> Path:
    """The checkout that owns the books: the one whose backend/data DATA is (the dev copy links its backend/data to
    the live one), else this one (DATA is a copy elsewhere, PRDL_DATA)."""
    found = DATA.resolve().parents[1]
    return found if (found / "backend").is_dir() and (found / "PRDL").is_dir() else ROOT


def duplicate_candidates(home: Path = ROOT) -> Path:
    """The candidate pairs of the last scan (duplicates/scan.py)."""
    return home / "backend" / "enrich" / "dubbelen_kandidaten.csv"


def duplicates_seen(home: Path = ROOT) -> Path:
    """The works whose text the last full scan read, with their PDF: what is new for the gate."""
    return home / "backend" / "enrich" / "dubbelen_gezien.txt"


def duplicate_decisions(home: Path = ROOT) -> Path:
    """The owner's decisions on the screen of duplicates/screen.py."""
    return home / "PRDL" / "dubbelen_besluiten.csv"


def held_duplicates(home: Path = ROOT) -> Path:
    """The fixed list of step 0: works the gate holds as duplicates when there are no candidates."""
    return home / "backend" / "transcription" / "vasthouden_dubbel.tsv"


def no_translate(home: Path = ROOT) -> Path:
    """Works translate.py does not translate: each is the same book as another work, whose reading edition is."""
    return home / "backend" / "translation" / "niet_vertalen.tsv"


def special_layout(home: Path = ROOT) -> Path:
    """Works with a layout the transcription prompt is not made for (columns, a table, notes beside the text)."""
    return home / "backend" / "transcription" / "special_layout.tsv"


# --- the claude command, for the subscription (common/via_claude.py) ---
CLAUDE_LOCAL = Path.home() / ".local" / "bin" / ("claude.exe" if os.name == "nt" else "claude")   # the native installer's
# The desktop app on Windows is an MSIX package: outside the app its %APPDATA% lives under Packages\Claude_*.
CLAUDE_DESKTOP = (r"%APPDATA%\Claude\claude-code\*\*\claude.exe",
                  r"%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude\claude-code\*\*\claude.exe")
