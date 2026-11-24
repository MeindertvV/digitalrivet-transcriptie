"""What a request to a model costs, and the client that sends it.

One place for everything about asking a model (REFACTOR_PLAN.md steps 58-61). The prices stood in
transcription/transcribe.py, from which the translation took cost_of too; client() is what transcribe.py,
translate.py, translate_toc.py, profile.py and genre_haiku.py each built themselves; is_haiku() is the rule of
translate.py. MODEL_HAIKU is the one name of the model that translates and transcribes (CLAUDE.md).
"""

MODEL_HAIKU = "claude-haiku-5-5"

# $ per million tokens (input, output). Cache writes cost 1.25x input, cache reads 0.1x.
PRICES = {
    MODEL_HAIKU: (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}


def cost_of(model: str, usage, batch: bool) -> float:
    p_in, p_out = PRICES.get(model, PRICES.get(model.rsplit("-", 1)[0], (0, 0)))
    get = lambda k: getattr(usage, k, None) or 0
    dollars = (get("input_tokens") * p_in
               + get("cache_creation_input_tokens") * p_in * 1.25
               + get("cache_read_input_tokens") * p_in * 0.1
               + get("output_tokens") * p_out) / 1e6
    return dollars / 2 if batch else dollars


def is_haiku(model: str) -> bool:
    """Whether a model name is Haiku, as `claude --model` or the API names it. Only Haiku translates (CLAUDE.md)."""
    return model.startswith("claude-haiku-") or model == "haiku"


def client(workspace: str | None = None, **options):
    """An Anthropic client. A key not scoped to a workspace must name one on every request: give it as workspace.
    The rest (max_retries, api_key) goes to the client as it is. anthropic is only loaded here, so a script that
    only counts costs does not load it."""
    import anthropic
    if workspace:
        options["default_headers"] = {"anthropic-workspace-id": workspace}
    return anthropic.Anthropic(**options)
