from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, TypedDict, TypeAlias

import polars as pl

FrameLike: TypeAlias = pl.DataFrame | pl.LazyFrame
DiagnosticReport: TypeAlias = Mapping[str, Mapping[str, Any]]
DefaultParams: TypeAlias = Dict[str, Any]
BasicInfoMetrics: TypeAlias = Dict[str, Dict[str, Any]]


class StrategyContext(TypedDict):
    all_columns: list[str]
    numeric_columns: list[str]
    schema: Mapping[str, pl.DataType]


StrategyExprs: TypeAlias = list[pl.Expr] | Dict[str, pl.Expr]
StrategyFn: TypeAlias = Callable[[FrameLike, StrategyContext], StrategyExprs]


@dataclass(frozen=True)
class OutputConfig:
    report_path: str | Path | None = None
    plot_dir: str | Path | None = None
    parameter_path: str | Path | None = None


@dataclass
class VisualizeConfig:
    dpi: int = 300
    size: tuple[int, int] = (10, 6)
    format: str = "pdf"
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "data" / "report")
    save_to_disk: bool = True
    overwrite: bool = True
    close_after_save: bool = True
    theme: str = "whitegrid"
    palette: str = "viridis"
    font_scale: float = 1.2
    font_family: str = "sans-serif"
    max_categories: int = 20
    tight_layout: bool = True
    show_kde: bool = True
