from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import RunMetadata


class AigcBackend(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> dict[str, Any]: ...

    @abstractmethod
    def invoke(self, operation: str, payload: dict[str, Any]) -> tuple[dict[str, Any], RunMetadata]: ...

    @abstractmethod
    def generation_run(self, kind: str, payload: dict[str, Any]) -> RunMetadata: ...

    @abstractmethod
    def generate(self, kind: str, payload: dict[str, Any], run: RunMetadata) -> dict[str, Any]: ...
