from __future__ import annotations

from typing import Any, Dict

from signaltwice.core.interface import BaseParameterStrategy
from signaltwice.core.types import DiagnosticReport
from signaltwice.engine.registry import parameters


@parameters.register_strategy("misvalue")
class MisvalueStrategy(BaseParameterStrategy):
    DROP_THRESHOLD: float = 50.0

    def __call__(self, diagnostic_report: DiagnosticReport) -> Dict[str, Any]:
        dtypes = self._get_section(diagnostic_report, "get_dtypes")
        nulls = self._get_section(diagnostic_report, "count_nulls")

        by_column: Dict[str, Dict[str, Any]] = {}
        for column, dtype in dtypes.items():
            null_percent = self._get_nested_float(nulls, column, "percent")

            if null_percent is None or null_percent <= 0:
                continue

            if null_percent >= self.DROP_THRESHOLD:
                method = "drop"
            elif self._is_numeric_dtype(str(dtype)):
                method = "mean"
            else:
                method = "mode"

            by_column[column] = {
                "method": method,
                "null_percent": null_percent,
            }

        return {"by_column": by_column}


@parameters.register_strategy("encode")
class EncodeStrategy(BaseParameterStrategy):
    ONEHOT_LIMIT: int = 10

    def __call__(self, diagnostic_report: DiagnosticReport) -> Dict[str, Any]:
        dtypes = self._get_section(diagnostic_report, "get_dtypes")
        uniques = self._get_section(diagnostic_report, "count_uniques")

        by_column: Dict[str, Dict[str, Any]] = {}
        for column, dtype in dtypes.items():
            if not self._is_categorical_dtype(str(dtype)):
                continue

            n_unique = uniques.get(column)
            unique_count = int(n_unique) if isinstance(n_unique, (int, float)) else None

            if unique_count is not None and unique_count < self.ONEHOT_LIMIT:
                method = "onehot"
            else:
                method = "label"

            by_column[column] = {
                "method": method,
                "n_unique": unique_count,
            }

        return {"by_column": by_column}


@parameters.register_strategy("normalize")
class NormalizeStrategy(BaseParameterStrategy):
    SKEWNESS_THRESHOLD: float = 1.0

    def __call__(self, diagnostic_report: DiagnosticReport) -> Dict[str, Any]:
        dtypes = self._get_section(diagnostic_report, "get_dtypes")
        distribution = self._get_section(diagnostic_report, "calc_distribution_shape")

        by_column: Dict[str, Dict[str, Any]] = {}
        for column, dtype in dtypes.items():
            if not self._is_numeric_dtype(str(dtype)):
                continue

            skewness = self._get_nested_float(distribution, column, "skewness")

            if skewness is not None and abs(skewness) > self.SKEWNESS_THRESHOLD:
                method = "robust"
            else:
                method = "standard"

            by_column[column] = {
                "method": method,
                "skewness": skewness,
            }

        return {"by_column": by_column}
