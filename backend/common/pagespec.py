"""The scans named on a command line: "20-24,30".

Seven scripts read this, each with a loop of its own. What they did with the numbers differed (a set, a count,
only those inside the book) and still does; the reading is here.
"""
import sys


def page_numbers(spec: str | None, skip_empty: bool = False) -> list[int]:
    """The scan numbers of a spec like "20-24,30", in the order given; a scan named twice comes twice.
    Something that is no number raises ValueError, also an empty part ("1,,3") unless skip_empty."""
    out: list[int] = []
    for part in (spec or "").split(","):
        if skip_empty and not part:
            continue
        a, _, b = part.strip().partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def parse_pages(spec: str | None, count: int) -> list[int]:
    """The scans of a book of `count` scans that a spec names, each once, in order; all of them without a spec.
    A scan outside the book stops the program: this is for a command line."""
    if not spec:
        return list(range(1, count + 1))
    pages = set(page_numbers(spec))
    bad = [p for p in pages if not 1 <= p <= count]
    if bad:
        sys.exit(f"Pages out of range 1-{count}: {sorted(bad)}")
    return sorted(pages)
