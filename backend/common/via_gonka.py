"""Send one request to a model on the Gonka network through an OpenAI-compatible broker.

Used by translate.py with `--via gonka` (default model zai-org/GLM-5.3-Flash). The key comes from
GONKA_API_KEY, else from keys/GonkaKey.txt (never put the key in a script); the broker
from GONKA_BASE_URL, else BASE_URL below. Text only: the broker refuses images. Returns an object
shaped like an API Message, so the callers can treat all routes alike; its cost is 0 here because
the broker bills its own credit.

The model always reasons before it answers, and the broker stops every answer at 4096 tokens,
reasoning included. Keep requests small (translate.py: --words 250) and the effort low.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

BASE_URL = "https://hskyauefqcgbvgvxkluj.supabase.co/functions/v1/gonka/v1"
MAX_TOKENS = 4096  # the broker's ceiling
STOP = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}
RETRY = {429, 500, 502, 503, 504}
NO_CREDIT = 402  # the broker's answer when the credit is spent


class NoCredit(RuntimeError):
    """The Gonka credit is spent: every later request fails the same way, so the run stops."""

_key = None
_key_lock = threading.Lock()


def key() -> str:
    global _key
    with _key_lock:
        if _key is None:
            from common.env import read_key
            _key = read_key("GONKA_API_KEY", "GonkaKey.txt")
            if not _key:
                raise RuntimeError("Gonka key not found: set GONKA_API_KEY or put it in keys/GonkaKey.txt")
    return _key


class Usage(SimpleNamespace):
    def model_dump(self):
        return dict(vars(self))


def ask(system: str, text: str, model: str, effort: str | None = "low", timeout: int = 600, tries: int = 4):
    """One user turn with `system` as the system prompt. Retries busy or failing broker answers;
    raises RuntimeError when they keep failing, and NoCredit at once when the credit is spent."""
    body = {"model": model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
    if effort:
        # The broker knows low/medium/high; the pipeline's higher levels mean "high" here.
        body["reasoning_effort"] = effort if effort in ("low", "medium", "high") else "high"
    base = os.environ.get("GONKA_BASE_URL", BASE_URL).rstrip("/")
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(base + "/chat/completions", method="POST", data=data,
                                     headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                answer = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            reason = f"gonka: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code == NO_CREDIT:
                raise NoCredit(reason)
            if e.code not in RETRY:
                raise RuntimeError(reason)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            reason = f"gonka: {type(e).__name__}: {e}"
        time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(reason)
    choice = answer["choices"][0]
    used = answer.get("usage") or {}
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=choice["message"].get("content") or "")],
        model=f"gonka/{answer.get('model') or model}",
        stop_reason=STOP.get(choice.get("finish_reason"), choice.get("finish_reason")),
        usage=Usage(input_tokens=used.get("prompt_tokens", 0), output_tokens=used.get("completion_tokens", 0),
                    reasoning_tokens=(used.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)),
    )
