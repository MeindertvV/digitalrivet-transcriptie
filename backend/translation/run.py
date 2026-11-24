"""Asking the model for a translation (REFACTOR_PLAN.md step 80: was the requests and the run modes of translate.py,
which still has the names).

user_message and complete send one request (through the API, the Claude Code CLI or Gonka); Run keeps what one run
does to a work's translation file (raw/, bad/, failed groups); run_sync asks group by group, a few at a time, and
translates a paragraph too long for one answer in pieces (translate_long) or a group that keeps failing unit by unit
(split_up); run_batch sends the groups as message batches and collects them, also on a later run (batches.json);
translate_running_heads does the running heads.
"""

import hashlib
import json
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from common import via_claude, via_gonka  # noqa: E402
from common.llm import cost_of  # noqa: E402
from common.units import Unit, balance, plain  # noqa: E402
from translation.store import Store  # noqa: E402
from translation.validate import foreign_script, parse_answer  # noqa: E402


# ---------------------------------------------------------------- requests

def tail_words(text: str, n: int, end: bool) -> str:
    words = plain(text).split()
    return " ".join(words[-n:] if end else words[:n])


def user_message(units: list[Unit], before: str, after: str, title: str, language: str) -> str:
    """The request for `units`, with the Latin just before and after them (if any) as context."""
    parts = [f"Chapter: {title}"]
    if before:
        parts.append(f"<context_before>…{tail_words(before, 60, True)}</context_before>")
    lines = [f"[{k}] {'## ' if u.heading else ''}{u.la}" for k, u in enumerate(units, 1)]
    parts.append("<passage>\n" + "\n".join(lines) + "\n</passage>")
    if after:
        parts.append(f"<context_after>{tail_words(after, 40, False)}…</context_after>")
    parts.append(f"Translate units [1] to [{len(units)}] into {language}, one line per unit.")
    return "\n\n".join(parts)


def build_params(args, system: str, text: str) -> dict:
    return dict(
        model=args.model,
        max_tokens=args.max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": text}],
        output_config={"effort": args.effort},
    )


def prompt_sha(system: str) -> str:
    """Which prompt wrote a translation: the system prompt as sent, glossary and book description
    included, since a file name or version does not say."""
    return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


def complete(client, args, system: str, text: str):
    """One request: through the API, or with --via claude through the Claude Code CLI (subscription)."""
    if args.via == "claude":
        return via_claude.ask(system, text, args.model, args.effort)
    if args.via == "gonka":
        return via_gonka.ask(system, text, args.model, args.effort)
    with client.messages.stream(**build_params(args, system, text)) as s:
        return s.get_final_message()


# ---------------------------------------------------------------- run

class Run:
    def __init__(self, work: Path, store: Store, groups: list[list[Unit]], titles: list[str], force: bool):
        self.raw, self.bad = work / "raw", work / "bad"
        for d in (self.raw, self.bad):
            d.mkdir(parents=True, exist_ok=True)
        self.work, self.store, self.groups, self.titles, self.force = work, store, groups, titles, force
        self.all = [u for g in groups for u in g]  # every unit in reading order, for the context
        self.pos = {id(u): k for k, u in enumerate(self.all)}
        self.lock = threading.Lock()
        self.spent = 0.0
        self.failed: dict[str, str] = {}
        self.fresh: set[str] = set()  # units translated in this run
        self.reqs: set[str] = set()  # raw/ files written in this run
        self.prompt_sha: str | None = None  # of the system prompt this run sends

    def gid(self, i: int) -> str:
        return self.groups[i][0].id

    def left(self, i: int) -> list[Unit]:
        """The units of group i still to translate in this run."""
        return [u for u in self.groups[i]
                if u.id not in self.fresh and (self.force or not self.store.done(u))]

    def neighbours(self, units: list[Unit]) -> tuple[str, str]:
        """The Latin of the units just before and just after `units` (empty at the ends)."""
        a, b = self.pos[id(units[0])], self.pos[id(units[-1])]
        return (self.all[a - 1].la if a > 0 else "",
                self.all[b + 1].la if b + 1 < len(self.all) else "")

    def message(self, i: int, units: list[Unit], language: str) -> str:
        return user_message(units, *self.neighbours(units), self.titles[i], language)

    def log(self, units: list[Unit], msg, model: str, batch: bool, prompt: str | None = None) -> tuple[str, float]:
        """Write raw/<req>.json for one answer (a second answer for the same first unit in this run
        gets <id>~2); returns req, which the units it translated keep, and the cost."""
        cost = cost_of(getattr(msg, "model", None) or model, msg.usage, batch)
        with self.lock:
            req, n = units[0].id, 1
            while req in self.reqs:
                n += 1
                req = f"{units[0].id}~{n}"
            self.reqs.add(req)
            self.spent += cost
        (self.raw / f"{req}.json").write_text(json.dumps({
            "units": [u.id for u in units], "model": msg.model, "prompt_sha256": prompt or self.prompt_sha,
            "stop_reason": msg.stop_reason, "batch": batch,
            "usage": msg.usage.model_dump(), "cost_usd": round(cost, 5),
        }, indent=2), encoding="utf-8")
        return req, cost

    def keep(self, units: list[Unit], translations: list[str], model: str, req: str):
        self.store.add(units, translations, model, req)
        with self.lock:
            self.fresh.update(u.id for u in units)

    def settle(self, i: int, reason: str, answer: str):
        """Group i is done, or still has units left: then `reason` and the answer go to failed/bad/."""
        gid = self.gid(i)
        with self.lock:
            if self.left(i):
                if reason:  # (a good answer for some units leaves the reason the others failed)
                    self.failed[gid] = reason
                    (self.bad / f"{gid}.txt").write_text(answer, encoding="utf-8")
            else:
                self.failed.pop(gid, None)
                (self.bad / f"{gid}.txt").unlink(missing_ok=True)

    def save(self, i: int, msg, model: str, batch: bool, units: list[Unit] | None = None,
             prompt: str | None = None) -> bool:
        """Validate one answer for `units` (default: all of group i) and store the units it
        translated well; `prompt`: the prompt_sha it was sent with, if not this run's. Returns
        True if all of them are good."""
        g = units or self.groups[i]
        req, cost = self.log(g, msg, model, batch, prompt)
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        good, bad = {}, {}
        problem = {"refusal": "refusal", "max_tokens": "hit max_tokens (raise --max-tokens)"}.get(msg.stop_reason)
        if not problem:
            good, bad, problem = parse_answer(text, g)
        if good:
            self.keep([g[k - 1] for k in sorted(good)], [good[k] for k in sorted(good)], msg.model, req)
        if not problem and bad:
            by_reason: dict[str, list[str]] = {}
            for k in sorted(bad):
                by_reason.setdefault(bad[k], []).append(g[k - 1].id)
            problem = "; ".join(f"unit(s) {ids[:10]} {r}" for r, ids in by_reason.items())
        self.settle(i, problem or "", text)
        status = "ok" if not problem else f"{len(good)} of {len(g)} units ok, {problem}" if good else f"FAILED: {problem}"
        details = getattr(msg.usage, "output_tokens_details", None)
        thinking = getattr(details, "thinking_tokens", None)
        print(f"  {req}: {status}  [{len(g)} units, {msg.usage.output_tokens} out"
              f"{f' ({thinking} thinking)' if thinking else ''}, ${cost:.4f}, total ${self.spent:.2f}]")
        return not problem

    def fail(self, i: int, reason: str):
        with self.lock:
            self.failed[self.gid(i)] = reason
        print(f"  {self.gid(i)}: FAILED: {reason}")


SENTENCE_END = re.compile(r"(?<=[.;:?!])\s+")


def too_long_for_gonka(args, u: Unit) -> bool:
    return args.via == "gonka" and u.words() > 3 * args.words


def translate_long(client, args, system, language, run: Run, i: int, u: Unit) -> bool:
    """A paragraph too long for one answer (Gonka stops every answer at 4096 tokens): cut it at
    sentence ends into pieces of about --words words, translate them in turn with their
    neighbours as context, and store the joined translation under the paragraph."""
    pieces, cur = [], ""
    for s in SENTENCE_END.split(u.la):
        if cur and len(plain(cur).split()) + len(plain(s).split()) > args.words:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        pieces.append(cur)
    cut = []
    for p in pieces:  # a "sentence" without punctuation that is still far too long
        words = p.split(" ")
        cut += [" ".join(words[k:k + args.words]) for k in range(0, len(words), args.words)] \
            if len(words) > 2 * args.words else [p]
    parts = [Unit(u.start[0], u.start[1], u.heading, p) for p in cut]
    balance(parts)
    before, after = run.neighbours([u])
    out, model, req = [], args.model, None
    for k, p in enumerate(parts):
        text = user_message([p], parts[k - 1].la if k else before,
                            parts[k + 1].la if k + 1 < len(parts) else after, run.titles[i], language)
        for _ in range(1 + args.retries):
            msg = complete(client, args, system, text)
            r, _cost = run.log([u], msg, args.model, False)
            req = req or r
            answer = "".join(b.text for b in msg.content if b.type == "text").strip()
            good, bad, problem = (({}, {}, "hit max_tokens") if msg.stop_reason == "max_tokens"
                                  else parse_answer(answer, [p]))
            if 1 in good:
                break
            problem = problem or bad.get(1)
        else:
            run.settle(i, f"{u.id}: piece {k + 1} of {len(parts)}: {problem}", answer)
            print(f"  {u.id}: FAILED in piece {k + 1} of {len(parts)}: {problem}")
            return False
        out.append(good[1])
        model = msg.model
    run.keep([u], [" ".join(out)], model, req)
    run.settle(i, "", "")
    print(f"  {u.id}: ok, {u.words()} words in {len(parts)} pieces")
    return True


def split_up(client, args, system, language, run: Run, i: int):
    """Translate the units of a group that keeps failing one by one, with the same context.
    Title pages need it: dozens of one-line units, where the model folds a fragment into the
    line before and leaves its own number empty."""
    for u in run.left(i):
        if too_long_for_gonka(args, u):
            translate_long(client, args, system, language, run, i, u)
            continue
        for _ in range(1 + args.retries):
            if run.save(i, complete(client, args, system, run.message(i, [u], language)), args.model, False, [u]):
                break


def run_sync(client, args, system, language, run: Run, todo: list[int]):
    def work(i):
        try:
            for _ in range(1 + args.retries):  # each attempt asks only for the units not yet good
                g = run.left(i)
                if not g:
                    return
                if len(g) == 1 and too_long_for_gonka(args, g[0]):
                    translate_long(client, args, system, language, run, i, g[0])
                    return
                run.save(i, complete(client, args, system, run.message(i, g, language)), args.model, False, g)
            if len(run.left(i)) > 1:
                print(f"  {run.gid(i)}: translating its {len(run.left(i))} remaining units one by one")
                split_up(client, args, system, language, run, i)
        except (anthropic.AuthenticationError, via_claude.UsageLimit, via_gonka.NoCredit):
            raise  # the same for every group: stop the run
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            run.fail(i, f"API error: {e}")
        except Exception as e:  # e.g. the connection dropped in the middle of a streamed answer
            run.fail(i, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(work, i) for i in todo]):
            f.result()


def custom_id(gid: str) -> str:
    return "g" + gid.replace(".", "_")  # batch ids allow letters, digits, _ and - only


def run_batch(client, args, system, language, run: Run, todo: list[int]):
    state_file = run.work / "batches.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else []
    index = {custom_id(run.gid(i)): i for i in range(len(run.groups))}
    pending = {cid for b in state for cid in b["groups"]}
    todo_new = [i for i in todo if custom_id(run.gid(i)) not in pending]
    if todo_new:
        sent = {custom_id(run.gid(i)): run.left(i) for i in todo_new}
        batch = client.messages.batches.create(requests=[
            Request(custom_id=custom_id(run.gid(i)), params=MessageCreateParamsNonStreaming(
                **build_params(args, system, run.message(i, sent[custom_id(run.gid(i))], language))))
            for i in todo_new])
        state.append({"id": batch.id, "groups": list(sent), "prompt": run.prompt_sha,
                      "units": {cid: [u.id for u in g] for cid, g in sent.items()}})
        state_file.write_text(json.dumps(state, indent=2))
        print(f"Submitted batch {batch.id} with {len(todo_new)} group(s)")

    while state:
        for b in list(state):
            info = client.messages.batches.retrieve(b["id"])
            if info.processing_status != "ended":
                c = info.request_counts
                print(f"Batch {b['id']}: {c.processing} processing, {c.succeeded} done, {c.errored} errored")
                continue
            for res in client.messages.batches.results(b["id"]):
                i = index.get(res.custom_id)
                if i is None:
                    continue  # the grouping changed since the batch was sent
                ids = b.get("units", {}).get(res.custom_id)  # the units sent (older state files: all)
                g = [u for u in run.groups[i] if ids is None or u.id in ids]
                if ids is not None and len(g) != len(ids):
                    run.fail(i, "units changed since the batch was sent")
                elif res.result.type == "succeeded":
                    run.save(i, res.result.message, args.model, True, g, prompt=b.get("prompt"))
                else:
                    run.fail(i, f"batch {res.result.type}")
            state.remove(b)
            state_file.write_text(json.dumps(state, indent=2))
        if state:
            time.sleep(args.poll)
    state_file.unlink(missing_ok=True)
    retry = [i for i in todo if run.left(i)]
    if retry:  # few and cheap: redo them right away without the batch discount, only the units left
        print(f"Retrying {len(retry)} group(s) with units left directly")
        run_sync(client, args, system, language, run, retry)


# ---------------------------------------------------------------- running heads

def running_heads(xml: Path) -> list[str]:
    heads = []
    for pg in ET.parse(xml).getroot().iter("page"):
        s = " ".join((pg.findtext("running_head") or "").split())
        if s.strip("[]").lower() not in ("", "none", "none printed"):
            heads.append(s)
    return list(dict.fromkeys(heads))


def translate_running_heads(client, args, system: str, todo: list[str]) -> tuple[dict[str, str], str]:
    """One request for all running heads not translated yet (several on Gonka, whose answers are short)."""
    if args.via == "gonka" and len(todo) > 40:
        found, model = {}, args.model
        for k in range(0, len(todo), 40):
            part, model = translate_running_heads(client, args, system, todo[k:k + 40])
            found.update(part)
        return found, model
    lines = "\n".join(f"[{k}] {t}" for k, t in enumerate(todo, 1))
    msg = complete(client, args, system, lines)
    text = "".join(b.text for b in msg.content if b.type == "text")
    found = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"^\s*\[(\d+)\]\s*(.+)$", text, re.M)}
    for k, t in list(found.items()):
        if reason := foreign_script(t):
            print(f"  running head [{k}] rejected: {reason}")
            del found[k]
    print(f"Translated {len(found)} of {len(todo)} running head(s)  [${cost_of(msg.model, msg.usage, False):.4f}]")
    return {t: found[k] for k, t in enumerate(todo, 1) if found.get(k)}, msg.model
