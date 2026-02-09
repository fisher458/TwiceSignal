from __future__ import annotations

import math
import re
from typing import Any, Mapping, Optional

import polars as pl
import polars.selectors as cs

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

from .types import FrameLike, StrategyContext

NESTED_SEP = "||"


def is_snake_case(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))


def ensure_lazy(frame: FrameLike) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    return frame.lazy()


def normalize_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def build_basic_info_context(frame: FrameLike) -> StrategyContext:
    lazy = ensure_lazy(frame)
    schema = lazy.collect_schema()
    all_columns = list(schema.keys())
    numeric_columns = list(lazy.select(cs.numeric()).collect_schema().keys())
    return {
        "all_columns": all_columns,
        "numeric_columns": numeric_columns,
        "schema": schema,
    }


def assign_nested(output: dict[str, Any], key: str, value: Any) -> None:
    if NESTED_SEP not in key:
        output[key] = normalize_value(value)
        return
    parts = key.split(NESTED_SEP)
    current: dict[str, Any] = output
    for part in parts[:-1]:
        next_bucket = current.get(part)
        if not isinstance(next_bucket, dict):
            next_bucket = {}
            current[part] = next_bucket
        current = next_bucket
    current[parts[-1]] = normalize_value(value)


class DataNormalizationMixin:
    def _to_pandas(self, data: Any) -> pd.Series | pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required for visualization utilities.")
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data
        if pl is not None and isinstance(data, (pl.Series, pl.DataFrame)):
            return data.to_pandas()
        if np is not None and isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.Series(data)
            if data.ndim == 2:
                return pd.DataFrame(data)
            raise ValueError("NumPy input must be 1D or 2D for visualization.")
        if isinstance(data, Mapping):
            if not data:
                return pd.DataFrame()
            values = list(data.values())
            if all(pd.api.types.is_scalar(value) or value is None for value in values):
                return pd.Series(data)
            return pd.DataFrame(data)
        if isinstance(data, (list, tuple)):
            if data and all(
                isinstance(item, (list, tuple, pd.Series))
                or (np is not None and isinstance(item, np.ndarray))
                for item in data
            ):
                return pd.DataFrame(data).T
            return pd.Series(list(data))
        raise ValueError("Unsupported data type for visualization.")

    def _to_series(self, data: Any) -> pd.Series:
        series = self._to_pandas(data)
        if isinstance(series, pd.DataFrame):
            if series.shape[1] != 1:
                raise ValueError("Expected a single series for visualization.")
            series = series.iloc[:, 0]
        return series.dropna()

    def _to_frame(self, data: Any) -> pd.DataFrame:
        frame = self._to_pandas(data)
        if isinstance(frame, pd.Series):
            frame = frame.to_frame()
        return frame.dropna(how="all")


class ReportStyler:
    _HEADER = "\033[95m"
    _BLUE = "\033[94m"
    _CYAN = "\033[96m"
    _GREEN = "\033[92m"
    _YELLOW = "\033[93m"
    _RED = "\033[91m"
    _BOLD = "\033[1m"
    _RESET = "\033[0m"
    REPORT_WIDTH = 62

    def __init__(self, enable_color: bool = True) -> None:
        self.enable_color = enable_color

    def _style(self, code: str) -> str:
        return code if self.enable_color else ""

    def header_color(self) -> str:
        return self._style(self._HEADER)

    def blue(self) -> str:
        return self._style(self._BLUE)

    def cyan(self) -> str:
        return self._style(self._CYAN)

    def green(self) -> str:
        return self._style(self._GREEN)

    def yellow(self) -> str:
        return self._style(self._YELLOW)

    def red(self) -> str:
        return self._style(self._RED)

    def bold(self) -> str:
        return self._style(self._BOLD)

    def reset(self) -> str:
        return self._style(self._RESET)

    def header(self, text: str) -> str:
        width = self.REPORT_WIDTH
        safe_text = text[:width]
        border = "═" * width
        content = f"{safe_text}".center(width)
        return (
            f"\n{self.cyan()}{self.bold()}╔{border}╗\n"
            f"║{content}║\n"
            f"╚{border}╝{self.reset()}"
        )

    def sub_header(self, text: str) -> str:
        return f"\n{self.bold()}>> {text}{self.reset()}"

    def key_value(
        self,
        key: str,
        value: Any,
        unit: str = "",
        pad: int = 25,
        color: str = "",
        icon: Optional[str] = None,
    ) -> str:
        val_str = f"{value}"
        if unit:
            val_str += f" {unit}"
        label = f"{key:<{pad}}"
        if icon:
            gutter = f"{icon} "
        else:
            gutter = "   "
        label = f"{gutter}{label}"
        return f"{label} : {color}{val_str}{self.reset()}"

    def warning_text(self, text: str) -> str:
        return f"{self.yellow()}⚠️  {text}{self.reset()}"

    def error_text(self, text: str) -> str:
        return f"{self.red()}❌ {text}{self.reset()}"

    def success_text(self, text: str) -> str:
        return f"{self.green()}✅ {text}{self.reset()}"

    def count_noun(self, count: int, singular: str, plural: str) -> str:
        return singular if count == 1 else plural

    def separator(self, text: str = "End of Report") -> list[str]:
        width = self.REPORT_WIDTH
        border_width = width + 2
        return [
            f"\n{self.cyan()}{'=' * border_width}{self.reset()}",
            f"{text.center(width)}",
            f"{self.cyan()}{'=' * border_width}{self.reset()}\n",
        ]


class SafeExtractor:
    @staticmethod
    def mapping(value: Any) -> dict:
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def mapping_from(container: Any, key: str, default: Any = None) -> dict:
        base = SafeExtractor.mapping(container)
        if default is None:
            default = {}
        return SafeExtractor.mapping(base.get(key, default))

    @staticmethod
    def deep_get(container: Any, path: list[Any], default: Any = None) -> Any:
        current = container
        for key in path:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
            if current is default:
                return default
        return current

    @staticmethod
    def deep_number(container: Any, path: list[Any], default: int = 0) -> int | float:
        return SafeExtractor.number(
            SafeExtractor.deep_get(container, path, default=default),
            default,
        )

    @staticmethod
    def deep_float(container: Any, path: list[Any], default: float = 0.0) -> float | None:
        value = SafeExtractor.deep_get(container, path, default=default)
        if value is None and default is None:
            return None
        return SafeExtractor.float(value, default)

    @staticmethod
    def deep_list(container: Any, path: list[Any], default: Any = None) -> list:
        if default is None:
            default = []
        return SafeExtractor.list(
            SafeExtractor.deep_get(container, path, default=default)
        )

    @staticmethod
    def truncate(text: Optional[str], max_len: int) -> str:
        if text is None:
            return ""
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return f"{text[:max_len - 3].rstrip()}..."

    @staticmethod
    def list(value: Any) -> list:
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def number(value: Any, default: int = 0) -> int | float:
        if isinstance(value, (int, float)):
            return value
        return default

    @staticmethod
    def float(value: Any, default: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return float(default)

    @staticmethod
    def describe_skewness(value: Any) -> str | None:
        if not isinstance(value, (int, float)):
            return None
        if value > 1:
            return "right-skewed"
        if value < -1:
            return "left-skewed"
        return "balanced"
