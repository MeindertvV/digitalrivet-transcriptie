"""Send one request through the Claude Code CLI (`claude -p`) instead of the API, so it runs on a
Claude subscription (Pro/Max) and counts against its usage limits instead of costing API money.

Used by transcribe.py and translate.py with `--via claude`. The CLI must be logged in once with
`claude auth login`. The API key is removed from the CLI's environment, otherwise the CLI would
bill the API after all. Returns an object shaped like an API Message, so the callers can treat both
routes alike; its cost is 0 because the subscription pays.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from common.llm import MODEL_HAIKU
from common.paths import CLAUDE_DESKTOP, CLAUDE_LOCAL

# Short names the CLI accepts; the pipeline's model ids map onto them.
ALIASES = {MODEL_HAIKU: "haiku", "claude-sonnet-5-5": "sonnet", "claude-opus-5-5": "opus"}
STOP = {"success": "end_turn", "error_max_turns": "max_tokens"}


def find_cli(local_first: bool = False) -> str:
    """The claude executable: $CLAUDE_CLI, then PATH, then the newest copy the desktop app installed. local_first:
    the native installer's place (common/paths.CLAUDE_LOCAL) before PATH when it is there, as interlinear/build.py
    wants it."""
    if os.environ.get("CLAUDE_CLI"):
        return os.environ["CLAUDE_CLI"]
    if local_first and CLAUDE_LOCAL.exists():
        return str(CLAUDE_LOCAL)
    found = shutil.which("claude")
    if found:
        return found
    copies = sorted((f for p in CLAUDE_DESKTOP for f in glob.glob(os.path.expandvars(p))), key=os.path.getmtime)
    if not copies:
        raise RuntimeError("claude CLI not found; set CLAUDE_CLI to its path")
    return copies[-1]


class Usage(SimpleNamespace):
    def model_dump(self):
        return dict(vars(self))


class UsageLimit(RuntimeError):
    """The subscription's usage limit is reached; later calls in this run fail at once."""


LIMIT_RE = re.compile(r"usage limit|hit your (\S+ )?limit|limit reached|rate.?limit|resets? (at|in)", re.I)
_limit_hit = threading.Event()


def call(args: list[str], content, *, cwd, timeout: int, cli: str | None = None, encoding: str | None = "utf-8",
         newline: bool = True) -> tuple[dict | None, subprocess.CompletedProcess]:
    """Start `claude -p <args>` (cli, else find_cli()) with one user turn on stdin (content: text or API content
    blocks; a newline after it unless newline=False), in the folder cwd, and read its stream: the last "result"
    event, or None, and the process. What the answer means (a failure, the usage limit) is for the caller.
    ask() below is the usual call; contents/toc_context.py gives its own command line (REFACTOR_PLAN.md step 60)."""
    # No ANTHROPIC_* (API key, base URL) and no CLAUDE* variables: started from inside the Claude
    # desktop app, the CLI would otherwise act as part of that app instead of using its own login.
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("ANTHROPIC", "CLAUDE"))}
    turn = {"type": "user", "message": {"role": "user", "content": content}}
    proc = subprocess.run([cli or find_cli(), "-p", *args], input=json.dumps(turn) + ("\n" if newline else ""),
                          capture_output=True, text=True, encoding=encoding, env=env, cwd=cwd, timeout=timeout)
    result = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:   # json.JSONDecodeError, the only one json.loads gives for a line of text
            continue
        if event.get("type") == "result":
            result = event
    return result, proc


def ask(system: str, content, model: str, effort: str | None = None, timeout: int = 900, *, cli: str | None = None):
    """One user turn (text, or a list of API content blocks such as an image plus text) with
    `system` as the whole system prompt and no tools. Raises RuntimeError on a CLI failure and
    UsageLimit once the subscription's limit is reached (then for every later call as well).
    cli: the claude executable, if not find_cli()'s."""
    if _limit_hit.is_set():
        raise UsageLimit("claude CLI: usage limit reached earlier in this run")
    with tempfile.TemporaryDirectory(prefix="via_claude_") as tmp:
        sys_file = Path(tmp) / "system.txt"
        sys_file.write_text(system, encoding="utf-8")
        args = ["--model", ALIASES.get(model, model),
                "--system-prompt-file", str(sys_file), "--tools", "", "--setting-sources", "",
                "--no-session-persistence", "--input-format", "stream-json",
                "--output-format", "stream-json", "--verbose"]
        if effort:
            args += ["--effort", effort]
        # An empty working folder, so the CLI picks up no project files or CLAUDE.md.
        result, proc = call(args, content, cli=cli, cwd=tmp, timeout=timeout)
    if result is None or result.get("is_error"):
        reason = (result or {}).get("result") or proc.stderr.strip()[-300:] or f"exit code {proc.returncode}"
        if LIMIT_RE.search(reason):
            _limit_hit.set()
            raise UsageLimit(f"claude CLI: {reason}")
        raise RuntimeError(f"claude CLI: {reason}")
    used = result.get("usage") or {}
    real_model = next(iter(result.get("modelUsage") or {}), ALIASES.get(model, model))
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=result.get("result") or "")],
        model=f"subscription/{real_model}",
        stop_reason=STOP.get(result.get("subtype"), result.get("subtype")),
        usage=Usage(input_tokens=used.get("input_tokens", 0), output_tokens=used.get("output_tokens", 0),
                    cache_creation_input_tokens=used.get("cache_creation_input_tokens", 0),
                    cache_read_input_tokens=used.get("cache_read_input_tokens", 0)),
    )
