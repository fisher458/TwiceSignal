from __future__ import annotations

from typing import Dict

import polars as pl

from signaltwice.core.types import StrategyContext
from signaltwice.core.utils import NESTED_SEP
from signaltwice.engine.registry import basic_info


@basic_info.register("get_shape")
def get_shape(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    return {
        "rows": pl.len(),
        "columns": pl.lit(len(context["all_columns"])),
    }


@basic_info.register("get_dtypes")
def get_dtypes(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    return [pl.lit(str(dtype)).alias(col) for col, dtype in context["schema"].items()]


@basic_info.register("get_column_names")
def get_column_names(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    return {"columns": pl.lit(context["all_columns"])}


@basic_info.register("count_nulls")
def count_nulls(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    cols = context["all_columns"]
    if not cols:
        return []

    rows_expr = pl.len()
    exprs = []
    for col in cols:
        exprs.append(pl.col(col).null_count().alias(f"{col}{NESTED_SEP}count"))
        exprs.append(
            pl.when(rows_expr == 0)
            .then(0.0)
            .otherwise(pl.col(col).null_count() / rows_expr * 100)
            .alias(f"{col}{NESTED_SEP}percent")
        )
    return exprs


@basic_info.register("count_uniques")
def count_uniques(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    cols = context["all_columns"]
    if not cols:
        return {}

    return {col: pl.col(col).n_unique() for col in cols}


@basic_info.register("calc_distribution_shape")
def calc_distribution_shape(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    numeric_cols = context["numeric_columns"]
    if not numeric_cols:
        return []

    exprs = []
    for col in numeric_cols:
        exprs.append(
            pl.col(col).skew().fill_nan(None).alias(f"{col}{NESTED_SEP}skewness")
        )
        exprs.append(
            pl.col(col).kurtosis().fill_nan(None).alias(f"{col}{NESTED_SEP}kurtosis")
        )
    return exprs
