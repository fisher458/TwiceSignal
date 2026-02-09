from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .types import BasicInfoMetrics, DefaultParams, DiagnosticReport, FrameLike


class ReaderService(Protocol):
    def auto_read(
        self,
        file_path: str | Path,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> FrameLike:
        ...


class WriterService(Protocol):
    def get_strategy(self, fmt: str) -> Any:
        ...

    def auto_write(self, content: Any, file_path: str | Path, **kwargs: Any) -> None:
        ...


class ParameterService(Protocol):
    def produce_default_parameters(
        self, diagnostic_report: DiagnosticReport | None = None
    ) -> DefaultParams:
        ...


class BasicInfoService(Protocol):
    def execute_all(self, frame: FrameLike) -> BasicInfoMetrics:
        ...


class ReportService(Protocol):
    def generate_full_report(
        self,
        raw_data: dict,
        execution_order: Sequence[str] | None = None,
        enable_color: bool = True,
    ) -> str:
        ...


class VisualizerService(Protocol):
    def run_batch(
        self,
        frame: Any,
        basic_info: BasicInfoMetrics,
        save_fn: Callable[[str, str, Any], None],
    ) -> None:
        ...
