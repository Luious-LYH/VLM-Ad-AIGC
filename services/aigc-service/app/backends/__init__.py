from .base import AigcBackend
from .fake import FakeBackend
from .local import LocalBackend


def create_backend(settings):
    return FakeBackend(settings) if settings.backend == "fake" else LocalBackend(settings)

__all__ = ["AigcBackend", "FakeBackend", "LocalBackend", "create_backend"]
