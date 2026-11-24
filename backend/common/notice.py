"""The scanner's own cover sheet in front of a book (e-rara: title data, "Persistent Link", terms of use in
four languages; HAB Wolfenbüttel: two pages of "Nutzungsbedingungen"). It is not part of the book, so no
reader, translation, download or register shows it.

The PDF keeps the sheet as scan 1, so scan numbers stay what they are: the page is only emptied.
"""

import re

NOTICE = re.compile(r"e-rara\.ch|e-rara-\d+|e-manuscripta"
                    r"|Nutzungsbedingungen für Online-Angebote|Wolfenbütteler Digitalen Bibliothek"
                    r"|Work originally published by HAB", re.I)
FIRST_SCANS = 2   # the sheet is the first scan; the OCR may put the next one in front of it


def notice_pages(root) -> set[int]:
    """Scan numbers of the cover sheets in a book.xml root (ElementTree or lxml)."""
    out = set()
    for page in root.iter("page"):
        n = page.get("n") or ""
        if n.isdigit() and int(n) <= FIRST_SCANS and NOTICE.search(" ".join("".join(page.itertext()).split())):
            out.add(int(n))
    return out


def blank_notice(root) -> set[int]:
    """Empty the cover sheets of a book.xml root in place (type="empty", no lines); returns their scans."""
    found = notice_pages(root)
    for page in root.iter("page"):
        if page.get("n", "").isdigit() and int(page.get("n")) in found:
            for child in list(page):
                page.remove(child)
            page.text = None
            page.set("type", "empty")
    return found
