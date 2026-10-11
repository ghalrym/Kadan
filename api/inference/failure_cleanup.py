"""Release allocations retained by completed failure frames."""
import traceback


def clear_failure_frames(error):
    """Release failed allocations retained anywhere in a chained loader exception."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend((current.__cause__, current.__context__))
        pending.extend(getattr(current, 'exceptions', ()))
