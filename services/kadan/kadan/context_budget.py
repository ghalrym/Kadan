"""Shared character heuristic used by the runner and context advice."""

MAX_CONTEXT = 48000
CHARACTERS_PER_TOKEN = 3
RESERVED_TOKENS = 1024


def context_characters(context_limit: int | None) -> int:
    if context_limit is None:
        return MAX_CONTEXT
    return max(0, (context_limit - RESERVED_TOKENS) * CHARACTERS_PER_TOKEN)
