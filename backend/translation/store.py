"""The translation file of a work (REFACTOR_PLAN.md step 79: was the class Store of translate.py, which still has
the name).

Store reads translation_<lang>.json, ties its units to the book as it is now (blocks.py), drops what is not in a
script a translation may use, and writes the file again after each answer that is kept.
"""

import json
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common.languages import TRANSLATED as LANGUAGES  # noqa: E402
from common.units import Unit  # noqa: E402
from translation import blocks  # noqa: E402
from translation.relink import KEEP, held  # noqa: E402
from translation.validate import foreign_script  # noqa: E402

class Store:
    """translation_<lang>.json: every translated unit, written again after each answer."""

    def __init__(self, book_dir: Path, lang: str, units: list[Unit]):
        self.path = book_dir / f"translation_{lang}.json"
        self.book, self.lang, self.units = book_dir.name, lang, units
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.blocks = blocks.sync(book_dir, units, False)["version"]
        before = data.get("blocks")
        linked = blocks.link(data, units, self.blocks)   # the units' ids are their place in the book as it is now
        if self.path.exists() and held(linked):   # the brake of relink.py: more lost than a cover sheet
            sys.exit(f"{self.book}: tying {self.path.name} to the book as it is now would retire {linked['retired']} "
                     f"translated unit(s); nothing written. Look with translation/relink.py {self.book}, and link with "
                     f"relink.py --write --force if that is right.")
        blocks.sync(book_dir, units)
        self.retired = data.get("retired", [])
        self.tr = {u["id"]: u for u in data.get("units", [])}
        self.invalid = {u["id"]: reason for u in self.tr.values()
                        if (reason := foreign_script(u.get("tr") or ""))}
        if self.invalid:
            print(f"{len(self.invalid)} translated unit(s) contain non-Latin text: translating them again")
        heads = data.get("running_heads", {})
        self.running_heads: dict[str, str] = {k: v for k, v in heads.items() if not foreign_script(v or "")}
        if len(self.running_heads) != len(heads):
            print(f"{len(heads) - len(self.running_heads)} running head(s) contain non-Latin text: "
                  f"translating them again")
        self.models = set(data.get("model", "").split(", ")) - {""}
        self.lock = threading.Lock()
        if linked["changed"]:
            print(f"{linked['changed']} unit(s) whose Latin changed since they were translated: translating them again")
        if linked["retired"]:
            print(f"{linked['retired']} unit(s) no longer in the book: kept in \"retired\", out of the reader")
        # Files written before units knew their lines (or after book.xml changed around them):
        # take start, end and lines from book.xml; costs nothing, the Latin is the same.
        self.relinked = 0
        for u in units:
            old = self.tr.get(u.id)
            if self.done(u) and (old.get("start"), old.get("end"), old.get("lines")) != (u.start, u.end, u.lines):
                self.tr[u.id] = {k: v for k, v in u.json(old["tr"], old.get("req")).items() if v is not None}
                self.relinked += 1
        if self.relinked:
            print(f"{self.relinked} unit(s) given their printed lines from book.xml")
        if self.path.exists() and (self.relinked or before != self.blocks or linked["retired"]):
            keep = book_dir / KEEP / self.path.name   # the translation as it was, once, as relink.py keeps it
            if not keep.exists():
                keep.parent.mkdir(exist_ok=True)
                keep.write_bytes(self.path.read_bytes())
            self.write()

    def done(self, u: Unit) -> bool:
        return (u.id in self.tr and self.tr[u.id]["la"] == u.la
                and u.id not in self.invalid)

    def add(self, units: list[Unit], translations: list[str], model: str, req: str):
        """Store translated units; `req`: the request (raw/<req>.json) that translated them."""
        with self.lock:
            for u, t in zip(units, translations):
                if reason := foreign_script(t):
                    self.invalid[u.id] = reason
                    self.tr.pop(u.id, None)
                    continue
                self.invalid.pop(u.id, None)
                self.tr[u.id] = u.json(t, req)
            self.models.add(model)
            self.write()

    def write(self):
        data = {"book": self.book, "lang": self.lang, "language": LANGUAGES[self.lang],
                "model": ", ".join(sorted(self.models)), "running_heads": self.running_heads, "blocks": self.blocks,
                "units": [self.tr[u.id] for u in self.units if u.id in self.tr and u.id not in self.invalid]}
        if self.retired:
            data["retired"] = self.retired
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
