from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from signaltwice.core.interface import BaseVisualizer
from signaltwice.engine.registry import visualizers


@visualizers.register_visualize_strategy("histogram")
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


@visualizers.register_visualize_strategy("barChart")
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
        indexed = [
            (idx, label, count) for idx, (label, count) in enumerate(zip(labels, counts))
        ]
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


@visualizers.register_visualize_strategy("boxPlot")
class BoxPlotVisualizer(BaseVisualizer):
    def __call__(self, data: Any, label: str, **kwargs: Any) -> plt.Figure:
        frame = self._to_frame(data)
        if frame.empty or frame.dropna(how="all").empty:
            return self._render_no_data(label, "Box Plot")

        fig, ax = self._create_canvas("Box Plot", label, ylabel=kwargs.pop("ylabel", "Value"))
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


@visualizers.register_visualize_strategy("scatterPlot")
class ScatterPlotVisualizer(BaseVisualizer):
    def __call__(self, data: Any, label: str | Tuple[str, str], **kwargs: Any) -> plt.Figure:
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
            dict.fromkeys([x_label, y_label, hue_label] if hue_label else [x_label, y_label])
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
