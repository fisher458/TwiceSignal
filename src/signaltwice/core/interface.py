from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt

from .types import DiagnosticReport, FrameLike
from .utils import DataNormalizationMixin


class BaseReader(ABC):
    @abstractmethod
    def __call__(
        self,
        file_path: str | Path,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> FrameLike:
        """Read data from a path and return a Polars frame."""

    def __repr__(self) -> str:  # pragma: no cover - tiny helper
        return f"<{self.__class__.__name__} Instance>"


class BaseWriter(ABC):
    def __call__(self, content: Any, file_path: str | Path, **kwargs: Any) -> None:
        path = Path(file_path)
        if path.parent and path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._write(content, path, **kwargs)

    @abstractmethod
    def _write(self, content: Any, path: Path, **kwargs: Any) -> None:
        """Write content to path."""

    def __repr__(self) -> str:  # pragma: no cover - tiny helper
        return f"<{self.__class__.__name__} Instance>"


class BaseParameterStrategy(ABC):
    @abstractmethod
    def __call__(self, diagnostic_report: DiagnosticReport) -> dict[str, Any]:
        """Generate default parameters from diagnostic report."""

    def _get_section(self, report: DiagnosticReport, key: str) -> Mapping[str, Any]:
        section = report.get(key, {})
        return section if isinstance(section, Mapping) else {}

    def _is_numeric_dtype(self, dtype: str | None) -> bool:
        if not dtype:
            return False
        dtype_lower = dtype.lower()
        return dtype_lower.startswith(("int", "uint", "float", "decimal"))

    def _is_categorical_dtype(self, dtype: str | None) -> bool:
        if not dtype:
            return False
        return dtype.lower() in {"string", "categorical", "enum", "utf8", "boolean", "object"}

    def _get_nested_float(
        self,
        section: Mapping[str, Any],
        column: str,
        key: str,
    ) -> float | None:
        value = None
        nested = section.get(column)
        if isinstance(nested, Mapping):
            value = nested.get(key)

        if isinstance(value, (int, float)):
            return float(value)
        return None


class BaseReportSection(ABC):
    @abstractmethod
    def calculate(self, raw_data: dict) -> dict:
        """Compute metrics for this report section."""

    @abstractmethod
    def render(self, metrics: dict, styler: Any) -> str:
        """Render metrics to text using the provided styler."""


class BaseVisualizer(DataNormalizationMixin, ABC):
    def __init__(self, config: Any) -> None:
        self.config = config

    def _create_canvas(
        self,
        title_prefix: str,
        label: str,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=self.config.size, dpi=self.config.dpi)
        ax.set_title(f"{title_prefix} of {label}", fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.35)
        if xlabel:
            ax.set_xlabel(xlabel, fontweight="medium")
        if ylabel:
            ax.set_ylabel(ylabel, fontweight="medium")
        return fig, ax

    def _render_no_data(self, label: str, title_prefix: str) -> plt.Figure:
        fig, ax = self._create_canvas(title_prefix, label)
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    @abstractmethod
    def __call__(self, data: Any, label: str, **kwargs: Any) -> Any:
        """Render a figure from input data."""
