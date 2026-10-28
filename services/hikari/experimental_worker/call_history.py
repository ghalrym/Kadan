"""The bounded text retained by a spoken call between turns."""


def append_call_history(messages: list[dict], role: str, text: str) -> list[dict]:
    history = list(messages)
    if text.strip():
        history.append({"role": role, "text": text[:4000]})
    while len(history) > 12 or sum(len(message["text"]) for message in history) > 12000:
        history.pop(0)
    return history
