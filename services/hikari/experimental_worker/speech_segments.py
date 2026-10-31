"""Shared script boundaries for native and experimental spoken replies."""

import re


def speech_segment(text: str, final=False) -> tuple[str, str]:
    """Bound buffered synthesis to a sentence or a short text segment."""
    boundary = re.search(r"[.!?\n](?:\s|$)", text)
    if boundary and boundary.end() <= 240:
        end = boundary.end()
    elif len(text) >= 240:
        end = text.rfind(" ", 0, 240)
        if end < 1:
            end = 240
    elif final:
        end = len(text)
    else:
        return "", text
    return text[:end].strip(), text[end:].lstrip()


def speech_scripts(text: str) -> list[str]:
    scripts = []
    while text.strip():
        script, text = speech_segment(text, final=True)
        if script:
            scripts.append(script)
    return scripts
