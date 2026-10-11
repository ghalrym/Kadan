"""Caller-visible inference failures and their HTTP status codes."""
class InferenceFailure(Exception):
    def __init__(self, detail: str, status_code: int = 503):
        """Attach an HTTP status to a caller-visible lifecycle or inference failure."""
        super().__init__(detail)
        self.status_code = status_code


class ResourceCancelled(RuntimeError):
    """Checkpoint preparation was cancelled before publication."""
