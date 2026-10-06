"""Explicitly unsupported tts adapter; no fake model or residency."""
from api.inference.feature import UnsupportedFeature


class TTSFeature(UnsupportedFeature):
    operations = ('generate',)
    name, workload = 'tts', 'tts'
    def __init__(self):
        super().__init__('No speech provider is configured. Speech generation and voice cloning are unavailable.')
    def select(self, request):
        return getattr(request, 'model', None)
    def validate(self, payload, operation):
        # Route-owned models import lazily to avoid the manager construction cycle.
        from api.routes.v1.audio.speech import SpeechRequest
        return SpeechRequest.model_validate(payload)
