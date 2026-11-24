"""The requests that read a page of a book (REFACTOR_PLAN.md step 73: was the start of the requests section of
transcribe.py, which still has the names).

A page goes to the model as one image, or as one image per part (parts.py), each with the transcription prompt as
a cached system text; call_sync sends one request and waits for the whole answer. prompt_sha names the prompt a
page was read with.
"""

import base64
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/: common/ and the other stages
from transcription import parts  # noqa: E402

# Models that get Anthropic's server-side refusal fallback in sync mode.
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"}


def prompt_sha(prompt: str) -> str:
    """Which prompt wrote a page: a version name or file name does not say, the text does."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def build_params(args, prompt: str, image: bytes, model: str, request: str = "Transcribe this page.") -> dict:
    return dict(
        model=model,
        max_tokens=args.max_tokens,
        system=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": base64.standard_b64encode(image).decode()}},
                {"type": "text", "text": request},
            ],
        }],
        output_config={"effort": args.effort},
    )


def page_requests(args, book, prompt: str, n: int, model: str) -> list[tuple[dict, int]]:
    """The requests that read page n, in reading order: one, or one per part (parts.py). Each with
    its size in the image's bytes."""
    plan = [] if args.whole_pages else book.parts(n)
    if not plan:
        img = book.render_jpeg(n, args.dpi, args.max_px)
        return [(build_params(args, prompt, img, model), len(img))]
    out = []
    for k, part in enumerate(plan, 1):
        img = book.render_part(n, part, args.dpi, args.max_px)
        out.append((build_params(args, prompt, img, model, parts.request_text(k, len(plan))), len(img)))
    return out


def call_sync(client, params: dict):
    if params["model"] in FALLBACK_MODELS:
        stream = client.beta.messages.stream(
            **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    else:
        stream = client.messages.stream(**params)
    with stream as s:
        return s.get_final_message()
