from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from signaltwice.core.types import BasicInfoMetrics, VisualizeConfig
from signaltwice.core.utils import DataNormalizationMixin
from signaltwice.engine.registry import VisualizerRegistry


class SmartPlotter(DataNormalizationMixin):
    def __init__(
        self,
        config: VisualizeConfig,
        registry: VisualizerRegistry,
    ) -> None:
        self.config = config
        self.registry = registry

    def iter_column_plots(
        self,
        frame: Any,
        basic_info: BasicInfoMetrics,
    ) -> Iterator[Tuple[str, str, Any]]:
        frame = self._to_frame(frame)
        columns = basic_info.get("get_column_names", {}).get("columns", [])
        if not columns and hasattr(frame, "columns"):
            columns = list(frame.columns)
        dtypes = basic_info.get("get_dtypes", {})
        nulls = basic_info.get("count_nulls", {})
        uniques = basic_info.get("count_uniques", {})
        dist_shape = basic_info.get("calc_distribution_shape", {})

        for column in columns:
            null_percent = (nulls.get(column) or {}).get("percent")
            if null_percent is not None and null_percent > 80:
                continue

            dtype = dtypes.get(column)
            n_unique = uniques.get(column)
            skewness = (dist_shape.get(column) or {}).get("skewness")
            series = frame[column]
            if dtype is None:
                dtype = series.dtype
            if n_unique is None:
                if hasattr(series, "nunique"):
                    n_unique = series.nunique(dropna=True)

            strategy_name = self._select_strategy(dtype, n_unique, skewness)
            if strategy_name is None:
                continue

            visualizer = self.registry.create_visualizer(strategy_name, self.config)
            data = frame[column]

            kwargs: Dict[str, Any] = {}
            if strategy_name == "histogram":
                if skewness is not None and abs(skewness) > 1.5:
                    if self._supports_log_scale(data):
                        kwargs["log_scale"] = True
            yield column, strategy_name, visualizer(data, column, **kwargs)

    def plot_correlations(
        self,
        frame: Any,
        pairs: Sequence[Tuple[str, str]],
        basic_info: BasicInfoMetrics | None = None,
    ) -> Iterator[Tuple[str, str, Any]]:
        frame = self._to_frame(frame)
        visualizer = self.registry.create_visualizer("scatterPlot", self.config)
        dtypes = basic_info.get("get_dtypes", {}) if basic_info else {}
        for x_col, y_col in pairs:
            if x_col not in frame.columns or y_col not in frame.columns:
                continue
            if dtypes:
                x_dtype = dtypes.get(x_col) or frame[x_col].dtype
                y_dtype = dtypes.get(y_col) or frame[y_col].dtype
                if not self._is_numeric(x_dtype) or not self._is_numeric(y_dtype):
                    continue
            data = frame.loc[:, [x_col, y_col]]
            label = (x_col, y_col)
            yield f"{x_col}_vs_{y_col}", "scatterPlot", visualizer(data, label)

    def run_batch(
        self,
        frame: Any,
        basic_info: BasicInfoMetrics,
        save_fn: Callable[[str, str, Any], None],
    ) -> None:
        for column, strategy, fig in self.iter_column_plots(frame, basic_info):
            save_fn(column, strategy, fig)
            if self.config.close_after_save:
                plt.close(fig)

    def _select_strategy(
        self,
        dtype: Any,
        n_unique: Any,
        skewness: Any,
    ) -> str | None:
        if self._is_categorical(dtype) or (
            isinstance(n_unique, (int, float)) and n_unique < 20
        ):
            return "barChart"
        if self._is_numeric(dtype) and (
            n_unique is None or isinstance(n_unique, (int, float)) and n_unique >= 20
        ):
            if skewness is not None and abs(skewness) >= 2:
                return "boxPlot"
            return "histogram"
        return None

    def _is_numeric(self, dtype: Any) -> bool:
        if dtype is None:
            return False
        if pd.api.types.is_numeric_dtype(dtype):
            return True
        value = str(dtype).lower()
        numeric_tokens = ("int", "float", "decimal", "numeric", "double")
        return any(token in value for token in numeric_tokens)

    def _is_categorical(self, dtype: Any) -> bool:
        if dtype is None:
            return False
        if isinstance(dtype, pd.CategoricalDtype):
            return True
        if pd.api.types.is_object_dtype(dtype):
            return True
        value = str(dtype).lower()
        categorical_tokens = (
            "categorical",
            "category",
            "object",
            "utf8",
            "string",
            "bool",
            "enum",
        )
        return any(token in value for token in categorical_tokens)

    def _supports_log_scale(self, data: Any) -> bool:
        series = self._to_series(data)
        if not pd.api.types.is_numeric_dtype(series):
            return False
        return (series > 0).all()
