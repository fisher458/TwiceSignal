from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Sequence, Tuple, Type

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from dataclasses import field

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


@dataclass
class VisualizeConfig:
    # --- 1. 基礎渲染 (I/O Properties) ---
    dpi: int = 300
    size: tuple[int, int] = (10, 6)
    format: str = "pdf"             # 批次報告建議用 png, 論文建議 pdf/svg
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "report")
    
    # --- 2. 執行邏輯 (Execution Flow) ---
    save_to_disk: bool = True       # 是否寫入硬碟
    overwrite: bool = True          # 是否覆蓋現有檔案
    close_after_save: bool = True   # 存完立即 plt.close() (記憶體救星)
    
    # --- 3. 視覺美學 (Aesthetics/Seaborn) ---
    theme: str = "whitegrid"        # Seaborn 主題: white, darkgrid, ticks 等
    palette: str = "viridis"        # 顏色調色盤
    font_scale: float = 1.2         # 字體縮放比例
    font_family: str = "sans-serif" # 如果有中文需求，需指定字體如 'Arial Unicode MS'
    
    # --- 4. 自動化限制 (Automation Constraints) ---
    max_categories: int = 20        # BarChart 最多顯示前幾名，避免 X 軸擠爆
    tight_layout: bool = True       # 自動調整邊距，防止標籤被切掉
    show_kde: bool = True



class VisualizeHandler:
    _visualize_strategies: Dict[str, Type["BaseVisualizer"]] = {}

    @classmethod
    def register_visualize_strategy(cls, name: str):
        def wrapper(visualizer_cls: Type["BaseVisualizer"]):
            cls._visualize_strategies[name] = visualizer_cls
            return visualizer_cls
        return wrapper
    
    @classmethod
    def create_visualizer(cls, name: str, config: VisualizeConfig) -> "BaseVisualizer":
        """根據名稱建立已初始化的視覺化實例"""
        if name not in cls._visualize_strategies:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return cls._visualize_strategies[name](config)
    

class DataNormalizationMixin:
    def _to_pandas(self, data: Any) -> pd.Series | pd.DataFrame:
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


class BaseVisualizer(DataNormalizationMixin, ABC):
    
    def __init__(self, config: VisualizeConfig):
        self.config = config

    def _create_canvas(
        self,
        title_prefix: str,
        label: str,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
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
    def __call__(self, data: Any, label: str, **kwargs: Any) -> plt.Figure:
        """Render a figure from input data while using stored config."""
        pass

@VisualizeHandler.register_visualize_strategy("histogram")
class HistogramVisualizer(BaseVisualizer):
    
    def __call__(self, data: Any, label: str, **kwargs: Any) -> plt.Figure:
        series = self._to_series(data)
        if series.empty:
            return self._render_no_data(label, "Distribution")

        fig, ax = self._create_canvas(
            "Distribution",
            label,
            xlabel=kwargs.pop("xlabel", "Value"),
            ylabel=kwargs.pop("ylabel", "Frequency"),
        )
        sns.histplot(
            series,
            bins=kwargs.pop("bins", "auto"),
            kde=kwargs.pop("kde", True),
            color=kwargs.pop("color", "#4c72b0"),
            alpha=kwargs.pop("alpha", 0.85),
            edgecolor=kwargs.pop("edgecolor", "white"),
            ax=ax,
            **kwargs,
        )
        return fig

@VisualizeHandler.register_visualize_strategy("barChart")
class BarChartVisualizer(BaseVisualizer):
    
    def __call__(self, data: Any, label: str, **kwargs: Any) -> plt.Figure:
        labels, counts = self._aggregate_data(data, kwargs.pop("pre_aggregated", False))
        if not labels:
            return self._render_no_data(label, "Bar Chart")

        fig, ax = self._create_canvas(
            "Bar Chart",
            label,
            xlabel=kwargs.pop("xlabel", "Category"),
            ylabel=kwargs.pop("ylabel", "Count"),
        )
        sns.barplot(
            x=labels,
            y=counts,
            color=kwargs.pop("color", "#55a868"),
            alpha=kwargs.pop("alpha", 0.85),
            ax=ax,
            **kwargs,
        )
        ax.tick_params(axis="x", rotation=kwargs.pop("rotation", 45))
        return fig
    
    def _aggregate_data(
        self,
        data: Any,
        pre_aggregated: bool,
    ) -> Tuple[Sequence[str], Sequence[float]]:
        if isinstance(data, Mapping):
            labels = [str(label) for label in data.keys()]
            counts = [float(count) for count in data.values()]
            return self._limit_categories(labels, counts)

        series = self._to_series(data)
        if pre_aggregated:
            labels = [str(label) for label in series.index]
            counts = [float(count) for count in series.values]
            return self._limit_categories(labels, counts)

        counter = Counter(series)
        labels = [str(label) for label in counter.keys()]
        counts = [float(count) for count in counter.values()]
        return self._limit_categories(labels, counts)

    def _limit_categories(
        self,
        labels: Sequence[str],
        counts: Sequence[float],
    ) -> Tuple[Sequence[str], Sequence[float]]:
        max_categories = self.config.max_categories
        if max_categories <= 0 or len(labels) <= max_categories:
            return list(labels), list(counts)

        keep_count = max_categories - 1
        indexed = [(idx, label, count) for idx, (label, count) in enumerate(zip(labels, counts))]
        indexed_sorted = sorted(indexed, key=lambda item: (-item[2], item[0]))

        if keep_count <= 0:
            total = sum(item[2] for item in indexed_sorted)
            return [self._others_label(labels)], [total]

        keep_indices = {idx for idx, _, _ in indexed_sorted[:keep_count]}
        kept_labels: list[str] = []
        kept_counts: list[float] = []
        others_total = 0.0
        for idx, label, count in indexed:
            if idx in keep_indices:
                kept_labels.append(label)
                kept_counts.append(count)
            else:
                others_total += count

        if others_total > 0 or len(kept_labels) < len(labels):
            kept_labels.append(self._others_label(labels))
            kept_counts.append(others_total)
        return kept_labels, kept_counts

    def _others_label(self, existing_labels: Sequence[str]) -> str:
        base_label = "Others"
        if base_label not in existing_labels:
            return base_label
        counter = 1
        candidate = f"{base_label} ({counter})"
        while candidate in existing_labels:
            counter += 1
            candidate = f"{base_label} ({counter})"
        return candidate

@VisualizeHandler.register_visualize_strategy("boxPlot")
class BoxPlotVisualizer(BaseVisualizer):
    
    def __call__(
        self,
        data: Any,
        label: str,
        **kwargs: Any,
    ) -> plt.Figure:
        frame = self._to_frame(data)
        if frame.empty or frame.dropna(how="all").empty:
            return self._render_no_data(label, "Box Plot")

        fig, ax = self._create_canvas(
            "Box Plot",
            label,
            ylabel=kwargs.pop("ylabel", "Value"),
        )
        long_frame = frame.melt(var_name="series", value_name="value").dropna()
        sns.boxplot(
            data=long_frame,
            x="series",
            y="value",
            palette=kwargs.pop("palette", "Set2"),
            ax=ax,
            **kwargs,
        )
        return fig

@VisualizeHandler.register_visualize_strategy("scatterPlot")
class ScatterPlotVisualizer(BaseVisualizer):
    
    def __call__(
        self,
        data: Any,
        label: str | Tuple[str, str],
        **kwargs: Any,
    ) -> plt.Figure:
        frame = self._to_frame(data)
        x_label: str | Any
        y_label: str | Any

        if isinstance(label, tuple):
            x_label, y_label = label
        elif isinstance(data, Mapping):
            data_keys = list(data.keys())
            if len(data_keys) >= 2:
                x_label, y_label = data_keys[0], data_keys[1]
            else:
                raise ValueError("ScatterPlotVisualizer requires at least two data series.")
        else:
            columns = list(frame.columns)
            if len(columns) < 2:
                raise ValueError("ScatterPlotVisualizer requires at least two data series.")
            x_label, y_label = columns[0], columns[1]

        if x_label not in frame.columns or y_label not in frame.columns:
            raise ValueError("ScatterPlotVisualizer requires x/y columns to exist.")
        hue = kwargs.pop("hue", None)
        hue_label: str | None = None
        if hue is not None:
            if isinstance(hue, str) and hue in frame.columns:
                hue_label = hue
            else:
                hue_frame = self._coerce_hue_frame(hue)
                if len(hue_frame) != len(frame):
                    raise ValueError("Hue data length must match x/y length.")
                if not hue_frame.index.equals(frame.index):
                    hue_label = hue_frame.columns[0]
                    hue_frame = pd.DataFrame(
                        hue_frame.iloc[:, 0].to_numpy(),
                        index=frame.index,
                        columns=[hue_label],
                    )
                hue_label = hue_frame.columns[0]
                if hue_label in frame.columns:
                    base_label = "hue_overlay"
                    candidate = base_label
                    counter = 1
                    while candidate in frame.columns:
                        candidate = f"{base_label}_{counter}"
                        counter += 1
                    hue_frame = hue_frame.rename(columns={hue_label: candidate})
                    hue_label = candidate
                frame = frame.join(hue_frame, how="left")

        drop_columns = list(
            dict.fromkeys(
                [x_label, y_label, hue_label] if hue_label else [x_label, y_label]
            )
        )
        frame = frame.loc[:, drop_columns].dropna(subset=drop_columns)
        if frame.empty:
            return self._render_no_data(
                label if isinstance(label, str) else "Scatter Plot",
                "Scatter Plot",
            )

        fig, ax = self._create_canvas(
            "Scatter Plot",
            label if isinstance(label, str) else f"{x_label} vs {y_label}",
            xlabel=kwargs.pop("xlabel", x_label),
            ylabel=kwargs.pop("ylabel", y_label),
        )
        if hue is not None:
            sns.scatterplot(
                data=frame,
                x=x_label,
                y=y_label,
                hue=hue_label,
                color=kwargs.pop("color", "#c44e52"),
                alpha=kwargs.pop("alpha", 0.85),
                ax=ax,
                **kwargs,
            )
        else:
            sns.scatterplot(
                data=frame,
                x=x_label,
                y=y_label,
                color=kwargs.pop("color", "#c44e52"),
                alpha=kwargs.pop("alpha", 0.85),
                ax=ax,
                **kwargs,
            )
        return fig

    def _coerce_hue_frame(self, hue: Any) -> pd.DataFrame:
        hue_data = self._to_pandas(hue)
        if isinstance(hue_data, pd.DataFrame):
            if hue_data.shape[1] != 1:
                raise ValueError("Expected a single hue series for scatter plot.")
            return hue_data
        return hue_data.to_frame(name="hue")


class SmartPlotter(DataNormalizationMixin):
    def __init__(
        self,
        config: VisualizeConfig,
        handler: Type[VisualizeHandler] = VisualizeHandler,
    ) -> None:
        self.config = config
        self.handler = handler

    def iter_column_plots(
        self,
        frame: Any,
        basic_info: Dict[str, Dict[str, Any]],
    ) -> Iterator[Tuple[str, str, plt.Figure]]:
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
                n_unique = series.nunique(dropna=True)

            strategy_name = self._select_strategy(dtype, n_unique, skewness)
            if strategy_name is None:
                continue

            visualizer = self.handler.create_visualizer(strategy_name, self.config)
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
        basic_info: Dict[str, Dict[str, Any]] | None = None,
    ) -> Iterator[Tuple[str, str, plt.Figure]]:
        frame = self._to_frame(frame)
        visualizer = self.handler.create_visualizer("scatterPlot", self.config)
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
        basic_info: Dict[str, Dict[str, Any]],
        save_fn: Callable[[str, str, plt.Figure], None],
    ) -> None:
        for column, strategy, fig in self.iter_column_plots(frame, basic_info):
            save_fn(column, strategy, fig)
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
        categorical_tokens = ("categorical", "category", "object", "utf8", "string", "bool", "enum")
        return any(token in value for token in categorical_tokens)

    def _supports_log_scale(self, data: Any) -> bool:
        series = self._to_series(data)
        if not pd.api.types.is_numeric_dtype(series):
            return False
        return (series > 0).all()
