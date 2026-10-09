"""Explicitly unsupported image adapter; no fake model or residency."""
from api.inference.request_execution import UnavailableRequestExecutor


class ImageRequests(UnavailableRequestExecutor):
    operations = ('generate', 'edit')
    name, workload = 'image', 'image'
    def __init__(self):
        super().__init__('No image provider is configured. Image generation and editing are unavailable.')
    def select(self, request):
        return getattr(request, 'model', None)
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.images.generations import ImageRequest
        from api.routes.v1.images.edits import ImageRequest as EditRequest
        return (EditRequest if operation == 'edit' else ImageRequest).model_validate(payload)
