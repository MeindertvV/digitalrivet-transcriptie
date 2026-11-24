"""What a transcription keeps on disk, page by page, and the book made from it (REFACTOR_PLAN.md step 75: was the
class Output of transcribe.py, which still has the names).

Output checks each answer and stores it: pages/pNNNN.xml (and pNNNN.para.xml, the page in paragraphs), the answer
that failed as pNNNN.xml.bad, raw/pNNNN.json with model and cost, failed.txt with the pages still to do. A run picks
up from there (transcribed, read_failed); resplit, long_s and revalidate work on the saved pages without a call.
merge makes book.xml and book.json, with diagrams.txt and no_ocr.txt beside them.
"""

import json
import re
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import bookfile  # noqa: E402
from common.bookfile import flag_merged_lines, mark_numbered  # noqa: E402
from common.llm import cost_of  # noqa: E402
from transcription import figures, latin_words, no_heading, split_lines, parts  # noqa: E402
from transcription.answer import parse_page  # noqa: E402
from transcription.merge import move_catchwords  # noqa: E402

DIAGRAM_MODEL = "claude-sonnet-5-5"   # for the pages in diagrams.txt: Haiku's first version of a diagram loses depth (prompt rule 9)
DIAGRAM_NOTE = re.compile(r"\bdiagram", re.I)


def tag(n: int) -> str:
    return f"p{n:04d}"


class Output:
    def __init__(self, root: Path, book: "Book | None" = None):
        self.root = root
        self.book = book
        self.pages = root / "pages"
        self.raw = root / "raw"
        for d in (self.pages, self.raw):
            d.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.spent = 0.0
        self.failed: dict[int, str] = {}

    def transcribed(self, n: int, pages_dir: Path | None = None) -> bool:
        """True if the page has lines. A page marked as missing has none and is tried again."""
        f = (pages_dir or self.pages) / f"{tag(n)}.xml"
        return f.exists() and not f.read_text(encoding="utf-8").lstrip().startswith('<page type="missing')

    def save_missing(self, n: int, reason: str, pages_dir: Path | None = None):
        """Write a page XML that says the page is not in the book, so book.json never leaves a
        page out without a trace: <page type="missing"> with a <warning> giving the reason.
        `blank`: the page holds no text (a blank leaf, a plate); anything else is a page that
        should have been transcribed and was not."""
        pages_dir = pages_dir or self.pages
        pages_dir.mkdir(parents=True, exist_ok=True)
        page = ET.Element("page", type="missing" if reason != "blank" else "blank")
        w = ET.Element("warning", type="blank" if reason == "blank" else "missing_page")
        w.text = (f"page {n} was not transcribed: the page holds no text" if reason == "blank"
                  else f"page {n} was not transcribed: {reason}")
        w.tail = "\n  "
        page.append(w)
        (pages_dir / f"{tag(n)}.xml").write_text(ET.tostring(page, encoding="unicode"), encoding="utf-8")

    def save(self, n: int, msg, requested_model: str, batch: bool, pages_dir: Path | None = None,
             prompt: str | None = None) -> bool:
        """Validate and store one response (`prompt`: prompt_sha of the prompt it was sent with).
        Returns True if the page is good."""
        pages_dir = pages_dir or self.pages
        raw_dir = pages_dir.parent / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        model = getattr(msg, "model", None) or requested_model
        cost = cost_of(model, msg.usage, batch)
        text = "".join(b.text for b in msg.content if b.type == "text")
        (raw_dir / f"{tag(n)}.json").write_text(json.dumps({
            "page": n, "model": model, "requested_model": requested_model, "prompt_sha256": prompt,
            "stop_reason": msg.stop_reason, "batch": batch,
            "usage": msg.usage.model_dump(), "cost_usd": round(cost, 5),
        }, indent=2), encoding="utf-8")

        if msg.stop_reason == "refusal":
            xml, problem, warning = None, "refusal", None
        elif msg.stop_reason == "max_tokens":
            xml, problem, warning = None, "hit max_tokens (raise --max-tokens)", None
        else:
            xml, problem, warning = parse_page(text)
        if xml and not problem:
            xml, warning = self.split(n, xml, warning, pages_dir)
            xml, w = parts.check(xml, self.book.ocr_text(n) if self.book else None)
            warning = "; ".join(x for x in (warning, w) if x) or None
        with self.lock:
            self.spent += cost
        self.store(n, text, xml, problem, pages_dir)
        status = f"FAILED: {problem}" if problem else f"ok (warning: {warning})" if warning else "ok"
        print(f"  page {n}: {status}  [{model}, ${cost:.4f}, total ${self.spent:.2f}]")
        return not problem

    def split(self, n: int, xml: str, warning: str | None, pages_dir: Path) -> tuple[str, str | None]:
        """A page transcribed in paragraphs: keep the model's version as pNNNN.para.xml and
        return it split into printed lines with the OCR layer (split_lines.py). Either way a heading
        that is a whole Bible verse or a whole paragraph becomes a paragraph (no_heading.py)."""
        if not re.search(r"<(p|head)\b", xml):
            (pages_dir / f"{tag(n)}.para.xml").unlink(missing_ok=True)   # from an earlier run
            return self.no_headings(xml), warning
        (pages_dir / f"{tag(n)}.para.xml").write_text(xml + "\n", encoding="utf-8")
        xml, w = split_lines.split_page(xml, self.book.ocr(n) if self.book else None)
        if 'type="figure"' in xml and self.book is not None and self.book.doc is not None:
            with self.book.lock:   # the pictures as PNGs beside the pages (figures.py)
                xml = figures.cut(self.book.doc[n - 1], self.book.text_page(n), xml, n, self.root)
        page = ET.fromstring(xml)
        w2 = flag_merged_lines(page)
        if w2:
            xml = ET.tostring(page, encoding="unicode")
        return self.no_headings(xml), "; ".join(x for x in (warning, w, w2) if x) or None

    def no_headings(self, xml: str) -> str:
        """The page with its headings that are no headings made paragraphs (no_heading.demote_page)."""
        page = ET.fromstring(xml)
        if not no_heading.demote_page(page, no_heading.is_commentary(self.root.name)):
            return xml
        return ET.tostring(page, encoding="unicode")

    def resplit(self):
        """Split every saved paragraph page (pNNNN.para.xml) into lines again, no API calls."""
        files = sorted(self.pages.glob("p*.para.xml"))
        for f in files:
            n = int(f.name[1:5])
            xml, warning = self.split(n, f.read_text(encoding="utf-8").strip(), None, self.pages)
            (self.pages / f"{tag(n)}.xml").write_text(xml + "\n", encoding="utf-8")
            if warning:
                print(f"  page {n}: {warning}")
        print(f"Split {len(files)} page(s) into lines again")

    def long_s(self):
        """Write every ſ in the saved pages (pNNNN.xml and pNNNN.para.xml) as s, no API calls."""
        changed = 0
        for f in sorted(self.pages.glob("p*.xml")) + sorted(self.pages.glob("p*.para.xml")):
            text = f.read_text(encoding="utf-8")
            if "ſ" in text:
                f.write_text(text.replace("ſ", "s"), encoding="utf-8")
                changed += 1
        print(f"Wrote ſ as s in {changed} file(s)")

    def store(self, n: int, text: str, xml: str | None, problem: str | None, pages_dir: Path):
        with self.lock:
            if problem:
                (pages_dir / f"{tag(n)}.xml.bad").write_text(text, encoding="utf-8")
                self.failed_here(n, problem, pages_dir)
            else:
                (pages_dir / f"{tag(n)}.xml").write_text(xml + "\n", encoding="utf-8")
                (pages_dir / f"{tag(n)}.xml.bad").unlink(missing_ok=True)
                self.failed.pop(n, None)
            self.write_failed()   # a run stopped halfway leaves a correct failed.txt

    def revalidate(self):
        """Re-check saved failed responses (.xml.bad) with the current rules, no API calls."""
        rescued = 0
        for bad in sorted(self.pages.glob("p*.xml.bad")):
            n = int(bad.name[1:5])
            if self.transcribed(n):
                continue
            text = bad.read_text(encoding="utf-8")
            raw = self.raw / f"{tag(n)}.json"
            stop = json.loads(raw.read_text(encoding="utf-8")).get("stop_reason") if raw.exists() else None
            if stop in ("refusal", "max_tokens"):
                continue
            xml, problem, warning = parse_page(text)
            if xml and not problem:
                xml, warning = self.split(n, xml, warning, self.pages)
            self.store(n, text, xml, problem, self.pages)
            if not problem:
                rescued += 1
                print(f"  page {n}: ok{f' (warning: {warning})' if warning else ''}")
        print(f"Revalidated: {rescued} page(s) now accepted")

    def fail(self, n: int, reason: str, pages_dir: Path | None = None):
        with self.lock:
            self.failed_here(n, reason, pages_dir or self.pages)
        print(f"  page {n}: FAILED: {reason}")

    def failed_here(self, n: int, reason: str, pages_dir: Path):
        """A page that failed in pages_dir. A good page from an earlier run stays (a forced redo that fails costs
        nothing); otherwise the mark keeps the page visible in book.json. Only a failure in the book's own pages/
        counts for failed.txt: a comparison (--compare) has its own folder."""
        if pages_dir == self.pages:
            self.failed[n] = reason
        if not self.transcribed(n, pages_dir):
            self.save_missing(n, reason, pages_dir)

    def write_failed(self):
        """failed.txt: the pages that are still not transcribed, with the reason. A page that a
        later run did transcribe drops out; every such page is marked in book.json as well
        (<page type="missing">), so a mark left by an earlier run is replaced once it is redone.
        A failure of this run on a page that has a good version is not listed: the good page stayed.
        A line that does not start with a number (a note by hand) is kept as it was, at the top."""
        path = self.root / "failed.txt"
        old, notes = {}, []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                num, _, why = line.partition("\t")
                if num.isdigit():
                    old[int(num)] = why
                elif line.strip():
                    notes.append(line)
        old.update(self.failed)
        for n in list(old):
            f = self.pages / f"{tag(n)}.xml"
            if f.exists() and not f.read_text(encoding="utf-8").startswith('<page type="missing'):
                del old[n]  # transcribed, by this run or an earlier one
        if old or notes:
            path.write_text("".join(f"{line}\n" for line in notes) + "".join(f"{n}\t{w}\n" for n, w in sorted(old.items())), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        if old:
            print(f"{len(old)} page(s) failed, listed in {path}")

    def read_failed(self) -> list[int]:
        path = self.root / "failed.txt"
        if not path.exists():
            return []
        return [int(l.split("\t")[0]) for l in path.read_text(encoding="utf-8").splitlines() if l[:1].isdigit()]

    def merge(self, source: "Book"):
        book = ET.Element("book", source=source.source.name)
        files = sorted(f for f in self.pages.glob("p*.xml") if not f.name.endswith(".para.xml"))
        unsplit = []   # transcribed without an OCR layer: one line per paragraph
        for f in files:
            n = int(f.stem[1:])
            page = ET.fromstring(f.read_text(encoding="utf-8"))
            for parent in list(page.iter()):   # a <page> the model wrote inside the page (a cover sheet): not a page
                for inner in [c for c in parent if c.tag == "page"]:
                    parent.remove(inner)
            page.set("n", str(n))
            if page.get("lines") == "none":
                unsplit.append(n)
            raw = self.raw / f"{f.stem}.json"
            if page.get("type") != "missing" and raw.exists():
                page.set("model", json.loads(raw.read_text(encoding="utf-8"))["model"])
            book.append(page)
        mark_numbered(book)
        move_catchwords(book)
        mended = latin_words.repair_book(book)   # cut words, an f for a long s, soft hyphens, entities
        if sum(mended.values()):
            print("Mended words: " + ", ".join(f"{k} {v}" for k, v in mended.items() if v))
        ET.indent(book)
        out = self.root / "book.xml"
        bookfile.write(book, self.root)
        print(f"Merged {len(files)} page(s) into {out} and book.json")
        # diagrams.txt: the pages whose notes mention a diagram and that DIAGRAM_MODEL has not done: Haiku's
        # first version is kept; --redo-diagrams improves them once you say so
        diagrams = [(int(pg.get("n")), (pg.findtext("notes") or "").split()) for pg in book
                    if pg.get("model") != DIAGRAM_MODEL and DIAGRAM_NOTE.search(pg.findtext("notes") or "")]
        listing = self.root / "diagrams.txt"
        if diagrams:
            listing.write_text("".join(f"{n}\t{' '.join(w)[:120]}\n" for n, w in diagrams), encoding="utf-8")
            print(f"{len(diagrams)} page(s) with a diagram, listed in {listing} (--redo-diagrams for a better one)")
        else:
            listing.unlink(missing_ok=True)
        # no_ocr.txt: the pages whose lines are paragraphs, to --resplit once they have an OCR layer
        listing = self.root / "no_ocr.txt"
        if unsplit:
            listing.write_text("".join(f"{n}\n" for n in unsplit), encoding="utf-8")
            print(f"{len(unsplit)} page(s) without an OCR layer, not split into lines: listed in {listing}")
        else:
            listing.unlink(missing_ok=True)
