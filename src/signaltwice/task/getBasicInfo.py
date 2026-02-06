from __future__ import annotations

import re
import math
from typing import Any, Callable, Dict, Mapping, TypedDict

import polars as pl
import polars.selectors as cs


class StrategyContext(TypedDict):
    all_columns: list[str]
    numeric_columns: list[str]
    schema: Mapping[str, pl.DataType]


StrategyExprs = list[pl.Expr] | Dict[str, pl.Expr]
StrategyFn = Callable[[pl.DataFrame | pl.LazyFrame, StrategyContext], StrategyExprs]


_NESTED_SEP = "||"


def _is_snake_case(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", value))


def _ensure_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame
    return frame.lazy()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _is_string_like(dtype: Any) -> bool:
    if dtype is None:
        return False
    value = str(dtype).lower()
    string_tokens = ("utf8", "string", "categorical", "enum", "object")
    return any(token in value for token in string_tokens)


def _is_categorical_like(dtype: Any) -> bool:
    if dtype is None:
        return False
    value = str(dtype).lower()
    categorical_tokens = ("categorical", "enum", "string", "utf8", "object", "bool")
    return any(token in value for token in categorical_tokens)


def _looks_like_id(column: str) -> bool:
    name = column.lower()
    return name == "id" or name.endswith("_id")


def _build_context(frame: pl.DataFrame | pl.LazyFrame) -> StrategyContext:
    lazy = _ensure_lazy(frame)
    schema = lazy.schema
    all_columns = list(schema.keys())
    numeric_columns = list(lazy.select(cs.numeric()).schema.keys())
    return {
        "all_columns": all_columns,
        "numeric_columns": numeric_columns,
        "schema": schema,
    }


def _assign_nested(output: Dict[str, Any], key: str, value: Any) -> None:
    if _NESTED_SEP not in key:
        output[key] = _normalize_value(value)
        return
    parts = key.split(_NESTED_SEP)
    current = output
    for part in parts[:-1]:
        next_bucket = current.get(part)
        if not isinstance(next_bucket, dict):
            next_bucket = {}
            current[part] = next_bucket
        current = next_bucket
    current[parts[-1]] = _normalize_value(value)


class BasicInfoHandler:
    _basicinfo_strategies: Dict[str, StrategyFn] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[StrategyFn], StrategyFn]:
        if not _is_snake_case(name):
            raise ValueError(f"Strategy name '{name}' must be snake_case.")

        def wrapper(func: StrategyFn) -> StrategyFn:
            if not _is_snake_case(func.__name__):
                raise ValueError(
                    f"Strategy function '{func.__name__}' must be snake_case."
                )
            cls._basicinfo_strategies[name] = func
            return func

        return wrapper

    @classmethod
    def get_strategy(cls, name: str) -> StrategyFn:
        if name not in cls._basicinfo_strategies:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return cls._basicinfo_strategies[name]

    #entry point =====

    @classmethod
    def execute_all(
        cls, frame: pl.DataFrame | pl.LazyFrame
    ) -> Dict[str, Dict[str, Any]]:
        context = _build_context(frame)
        outputs: Dict[str, Dict[str, Any]] = {}
        exprs: list[pl.Expr] = []
        strategy_keys: Dict[str, list[str]] = {}

        for name, strategy in cls._basicinfo_strategies.items():
            strategy_exprs = strategy(frame, context)
            outputs[name] = {}
            if not strategy_exprs:
                continue
            keys: list[str] = []
            if isinstance(strategy_exprs, dict):
                for key, expr in strategy_exprs.items():
                    alias = f"{name}___{key}"
                    exprs.append(expr.alias(alias))
                    keys.append(key)
            else:
                for expr in strategy_exprs:
                    try:
                        key = expr.meta.output_name()
                    except Exception:
                        key = expr.meta.root_names()[0]
                    alias = f"{name}___{key}"
                    exprs.append(expr.alias(alias))
                    keys.append(key)
            strategy_keys[name] = keys

        if not exprs:
            return outputs

        result = _ensure_lazy(frame).select(exprs).collect()
        row = result.row(0, named=True)

        for name, keys in strategy_keys.items():
            strategy_output = outputs[name]
            for key in keys:
                alias = f"{name}___{key}"
                _assign_nested(strategy_output, key, row.get(alias))
        return outputs
    
    #entry point =====


@BasicInfoHandler.register("get_shape")
def get_shape(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    return {
        "rows": pl.len(),
        "columns": pl.lit(len(context["all_columns"])),
    }


@BasicInfoHandler.register("get_dtypes")
def get_dtypes(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    return [
        pl.lit(str(dtype)).alias(col)
        for col, dtype in context["schema"].items()
    ]


@BasicInfoHandler.register("get_column_names")
def get_column_names(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    return {"columns": pl.lit(context["all_columns"])}


@BasicInfoHandler.register("count_nulls")
def count_nulls(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    cols = context["all_columns"]
    if not cols:
        return []

    rows_expr = pl.len()
    exprs = []
    for col in cols:
        # 展開 cs.all() 為明確的 pl.col(col)
        exprs.append(
            pl.col(col).null_count().alias(f"{col}{_NESTED_SEP}count")
        )
        exprs.append(
            pl.when(rows_expr == 0)
            .then(0.0)
            .otherwise(pl.col(col).null_count() / rows_expr * 100)
            .alias(f"{col}{_NESTED_SEP}percent")
        )
    return exprs

@BasicInfoHandler.register("count_uniques")
def count_uniques(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    cols = context["all_columns"]
    if not cols:
        return {}

    return {col: pl.col(col).n_unique() for col in cols}


@BasicInfoHandler.register("calc_distribution_shape")
def calc_distribution_shape(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    numeric_cols = context["numeric_columns"]
    if not numeric_cols:
        return []

    exprs = []
    for col in numeric_cols:
        # 展開 cs.numeric()
        exprs.append(
            pl.col(col).skew().fill_nan(None).alias(f"{col}{_NESTED_SEP}skewness")
        )
        exprs.append(
            pl.col(col).kurtosis().fill_nan(None).alias(f"{col}{_NESTED_SEP}kurtosis")
        )
    return exprs


@BasicInfoHandler.register("get_descriptive_stats")
def get_descriptive_stats(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    numeric_cols = context["numeric_columns"]
    if not numeric_cols:
        return []

    exprs = []
    for col in numeric_cols:
        # 展開 cs.numeric()
        exprs.append(pl.col(col).min().fill_nan(None).alias(f"{col}{_NESTED_SEP}min"))
        exprs.append(pl.col(col).max().fill_nan(None).alias(f"{col}{_NESTED_SEP}max"))
        exprs.append(pl.col(col).mean().fill_nan(None).alias(f"{col}{_NESTED_SEP}mean"))
        exprs.append(pl.col(col).std().fill_nan(None).alias(f"{col}{_NESTED_SEP}std"))
    return exprs


@BasicInfoHandler.register("detect_outliers")
def detect_outliers(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> list[pl.Expr]:
    numeric_cols = context["numeric_columns"]
    if not numeric_cols:
        return []

    exprs: list[pl.Expr] = []
    for col in numeric_cols:
        q1 = pl.col(col).quantile(0.25)
        q3 = pl.col(col).quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (iqr * 1.5)
        upper = q3 + (iqr * 1.5)
        mask = ((pl.col(col) < lower) | (pl.col(col) > upper)).fill_null(False)
        exprs.append(
            mask.sum().fill_null(0)
            .alias(f"{col}{_NESTED_SEP}outlier_count")
        )
    return exprs


@BasicInfoHandler.register("check_duplication")
def check_duplication(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    # Performance note: struct(cs.all()) builds a row-wise struct for full-row
    # uniqueness. This can be expensive on wide tables. Use selectively.
    cols = context["all_columns"]
    if not cols:
        return {"duplicate_rows": pl.lit(0)}
    return {"duplicate_rows": pl.len() - pl.struct(cs.all()).n_unique()}


@BasicInfoHandler.register("check_data_health")
def check_data_health(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    cols = context["all_columns"]
    if not cols:
        return {
            "empty_rows_count": pl.lit(0),
            "empty_rows_percent": pl.lit(0.0),
        }

    null_exprs = [pl.col(col).is_null() for col in cols]
    nulls_per_row = pl.sum_horizontal(null_exprs)
    threshold = len(cols) * 0.5
    empty_mask = nulls_per_row > threshold
    empty_rows_count = empty_mask.sum()
    empty_rows_percent = (
        pl.when(pl.len() == 0)
        .then(0.0)
        .otherwise((empty_rows_count / pl.len()) * 100)
    )
    return {
        "empty_rows_count": empty_rows_count,
        "empty_rows_percent": empty_rows_percent,
    }


@BasicInfoHandler.register("check_rare_categories")
def check_rare_categories(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    exprs: Dict[str, pl.Expr] = {}
    for col, dtype in context["schema"].items():
        key = f"{col}{_NESTED_SEP}rare_category_count"
        # Target both string-like and categorical columns.
        if not _is_categorical_like(dtype):
            exprs[key] = pl.lit(0)
            continue
        rare_count = (
            pl.col(col)
            .value_counts(sort=False)
            .implode()
            .list.eval(
                pl.element().struct.field("count")
                < (pl.element().struct.field("count").sum() * 0.01)
            )
            .list.sum()
            .fill_null(0)
        )
        exprs[key] = rare_count
    return exprs


@BasicInfoHandler.register("check_string_variants")
def check_string_variants(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    exprs: Dict[str, pl.Expr] = {}
    for col, dtype in context["schema"].items():
        key = f"{col}{_NESTED_SEP}variant_noise_count"
        if not _is_string_like(dtype):
            exprs[key] = pl.lit(0)
            continue
        normalized = (
            pl.col(col)
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
        )
        raw_unique = pl.col(col).n_unique()
        normalized_unique = normalized.n_unique()
        diff = raw_unique - normalized_unique
        exprs[key] = pl.when(diff < 0).then(0).otherwise(diff).fill_null(0)
    return exprs


@BasicInfoHandler.register("check_constant_columns")
def check_constant_columns(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    exprs: Dict[str, pl.Expr] = {}
    for col in context["all_columns"]:
        key = f"{col}{_NESTED_SEP}is_constant"
        exprs[key] = pl.col(col).n_unique() <= 1
    return exprs


@BasicInfoHandler.register("check_id_integrity")
def check_id_integrity(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    id_columns = [col for col in context["all_columns"] if _looks_like_id(col)]
    if not id_columns:
        return {}

    # Performance note: per-ID struct(cs.all()) window scans can be heavy on large
    # tables. Consider narrowing id_columns or pre-filtering when necessary.
    exprs: Dict[str, pl.Expr] = {}
    for col in id_columns:
        key = f"{col}{_NESTED_SEP}inconsistent_row_count"
        inconsistent = pl.struct(cs.all()).n_unique().over(col).gt(1)
        exprs[key] = inconsistent.sum()
    return exprs

@BasicInfoHandler.register("get_samples")
def get_samples(
    frame: pl.DataFrame | pl.LazyFrame, context: StrategyContext
) -> Dict[str, pl.Expr]:
    if not context["all_columns"]:
        return {"samples": pl.lit([])}

    samples = pl.concat_list(
        [
            pl.struct(cs.all()).head(5).implode(),
            pl.struct(cs.all()).tail(5).implode(),
        ]
    ).list.flatten().list.unique()
    return {"samples": samples}
