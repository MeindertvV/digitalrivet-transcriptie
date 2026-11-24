"""Reading the model's answer to a page (REFACTOR_PLAN.md step 74: was the start of the output section of
transcribe.py, which still has the names).

parse_page takes the <page> XML out of an answer and repairs what the model often gets wrong: a bare & or <, a
remark where page furniture was left out, Cyrillic look-alikes, a missing </l>, italics that run over a closing
tag. join_parts makes the answers to the parts of a page (parts.py) one answer.
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.bookfile import INLINE, flag_merged_lines  # noqa: E402
from common.via_claude import Usage  # noqa: E402
from transcription import parts  # noqa: E402

# A bare "&" (as in Latin "&") is not valid XML; escape it so the page parses.
BARE_AMP = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")
# Page furniture the model filled in with a remark instead of leaving it out.
LOOKALIKES = str.maketrans("аеёосрхуіјАВЕКМНОРСТХ", "aeëocpxyijABEKMHOPCTX")  # Cyrillic for Latin; no book here has Russian
NO_VALUE = re.compile(r"<(page_number|running_head|catchword|signature)>\s*\[?(?:none(?: printed)?|not printed|n/?a)\]?\.?\s*</\1>\s*", re.I)
# A "<" that starts no tag, e.g. a library shelf mark "<36620148320015" on a bookplate.
BARE_LT = re.compile(r"<(?![A-Za-z_/!?])")


def join_parts(msgs: list, model: str):
    """The answers to the parts of a page as one answer, shaped like an API Message. An answer that
    cannot be read leaves the text as it is: save() then reports the page as failed."""
    texts = ["".join(b.text for b in m.content if b.type == "text") for m in msgs]
    usage = Usage(**{k: sum(getattr(m.usage, k, None) or 0 for m in msgs)
                     for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")})
    stop = next((m.stop_reason for m in msgs if m.stop_reason != "end_turn"), "end_turn")
    pages = [parse_page(t) for t in texts]
    text = "\n".join(texts)
    if stop == "end_turn" and all(xml and not problem for xml, problem, _ in pages):
        text = parts.join([ET.fromstring(xml) for xml, _, _ in pages])
    return SimpleNamespace(model=getattr(msgs[0], "model", None) or model, usage=usage, stop_reason=stop,
                           content=[SimpleNamespace(type="text", text=text)])


def parse_page(text: str) -> tuple[str | None, str | None, str | None]:
    """Extract the <page> XML from a model response. Returns (xml, problem, warning)."""
    m = re.search(r"<page\b[^>]*/>|<page\b.*</page>", text, re.S)
    # the model sometimes leaves out the <page> around its elements
    bare = None if m else re.search(r"<(?:layout|page_number|running_head|bottom_margin|main_text)\b.*(?:</notes>|</main_text>)", text, re.S)
    if not m and not bare:
        return None, "no <page>...</page> in response", None
    whole = m.group(0) if m else f'<page type="text">{bare.group(0)}</page>'
    xml = escape_notes(BARE_LT.sub("&lt;", BARE_AMP.sub("&amp;", NO_VALUE.sub("", whole.replace("<?>", "[?]"))))).replace("ſ", "s")  # the prompt asks for s
    xml = xml.replace("\u00ad", "-").translate(LOOKALIKES)  # a soft hyphen is the printed hyphen of a broken word
    repaired = False
    try:
        page = ET.fromstring(xml)
    except ET.ParseError as e:
        for fix in (close_lines, balance_tags, lambda x: balance_tags(close_lines(x))):
            try:  # a merged paragraph often lacks its closing </l>; italics often run over a </p>
                page = ET.fromstring(fix(xml))
                repaired = True
                break
            except ET.ParseError:
                pass
        else:
            return None, f"invalid XML: {e}", None
    warnings = [w for w in (flag_merged_lines(page),) if w]
    if repaired:
        note = "missing or unbalanced tag(s) repaired by the script; the words are unchanged"
        w = ET.Element("warning", type="repaired_xml")
        w.text, w.tail = note, "\n  "
        page.insert(0, w)
        warnings.append(note)
    if warnings:
        xml = ET.tostring(page, encoding="unicode")
    return xml, None, "; ".join(warnings) or None


def balance_tags(xml: str) -> str:
    """Repair the nesting of the tags: an element left open is closed where the element around
    it ends (an <i>/<hi>/<bl> closed there is opened again at the next text in that element), and a
    closing tag that closes nothing is dropped."""
    out, stack, pending = [], [], []
    for tok in re.split(r"(<[^>]+>)", xml):
        m = re.fullmatch(r"<(/?)([A-Za-z_]\w*)[^>]*?(/?)>", tok)
        if not m:
            if pending and tok.strip():   # italics that ran over a nested element resume here
                out.extend(f"<{t}>" for t in pending)
                stack.extend(pending)
                pending = []
            out.append(tok)
            continue
        close, name, empty = m.groups()
        if empty or tok.startswith(("<?", "<!")):
            out.append(tok)
        elif not close:
            if name not in INLINE:
                pending = []
            stack.append(name)
            out.append(tok)
        elif name in stack:
            while stack[-1] != name:
                t = stack.pop()
                out.append(f"</{t}>")
                if t in INLINE and name not in ("p", "head", "l", "main_text", "note"):
                    pending.append(t)
            stack.pop()
            out.append(tok)
        elif name in pending:
            pending.remove(name)
    return "".join(out)


def escape_notes(xml: str) -> str:
    """<notes> is free text: a "<l>" the model mentions there is not a tag."""
    return re.sub(r"(<notes>)(.*?)(</notes>)",
                  lambda m: m.group(1) + m.group(2).replace("<", "&lt;").replace(">", "&gt;") + m.group(3),
                  xml, flags=re.S)


def close_lines(xml: str) -> str:
    """Add a closing </l> to every <l> in <main_text> that lacks one."""
    m = re.search(r"<main_text>(.*?)</main_text>", xml, re.S)
    if not m:
        return xml
    parts = re.split(r"(?=<l\b)", m.group(1))
    fixed = [p if not p.startswith("<l") or "</l>" in p else p.rstrip() + "</l>\n    " for p in parts]
    return xml[:m.start(1)] + "".join(fixed) + xml[m.end(1):]
