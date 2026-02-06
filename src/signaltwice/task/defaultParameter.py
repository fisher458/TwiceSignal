from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Mapping

# Re-export for backward compatibility
from signaltwice.task.getBasicInfo import BasicInfoHandler

DiagnosticReport = Mapping[str, Mapping[str, Any]]
DefaultParams = Dict[str, Any]


class BaseParameterStrategy(ABC):
    """
    所有參數生成策略的基類。
    提供共用的報告解析方法 (Helper Methods)。
    """
    
    @abstractmethod
    def __call__(self, diagnostic_report: DiagnosticReport) -> Dict[str, Any]:
        """執行策略，根據報告生成參數。"""
        pass

    # --- Shared Utilities (工具方法) ---
    
    def _get_section(self, report: DiagnosticReport, key: str) -> Mapping[str, Any]:
        section = report.get(key, {})
        return section if isinstance(section, Mapping) else {}

    def _is_numeric_dtype(self, dtype: str | None) -> bool:
        if not dtype:
            return False
        # 改進 1: 轉小寫以兼容 Polars (Int64), Pandas (int64), Arrow 等不同風格
        dtype_lower = dtype.lower()
        return dtype_lower.startswith(("int", "uint", "float", "decimal"))

    def _is_categorical_dtype(self, dtype: str | None) -> bool:
        if not dtype:
            return False
        # 改進 1: 同樣轉小寫處理，增加相容性
        return dtype.lower() in {"string", "categorical", "enum", "utf8", "boolean", "object"}

    def _get_nested_float(self, section: Mapping[str, Any], column: str, key: str) -> float | None:
        value = None
        nested = section.get(column)
        if isinstance(nested, Mapping):
            value = nested.get(key)
        
        if isinstance(value, (int, float)):
            return float(value)
        return None


class ParameterHandler:
    _registry: ClassVar[Dict[str, type[BaseParameterStrategy]]] = {}
    _instances: ClassVar[Dict[type[BaseParameterStrategy], BaseParameterStrategy]] = {}

    @classmethod
    def register_strategy(cls, name: str):
        def wrapper(strategy_cls: type[BaseParameterStrategy]):
            if not issubclass(strategy_cls, BaseParameterStrategy):
                raise TypeError(f"{strategy_cls.__name__} must inherit from BaseParameterStrategy")
            
            # 改進 2: 防禦性檢查，避免意外覆蓋既有的策略
            if name in cls._registry:
                existing_cls = cls._registry[name].__name__
                raise ValueError(f"Strategy '{name}' is already registered by '{existing_cls}'.")
            
            cls._registry[name] = strategy_cls
            return strategy_cls
        return wrapper

    @classmethod
    def produce_default_parameters(
        cls, diagnostic_report: DiagnosticReport | None = None
    ) -> DefaultParams:
        """Generate default parameters from a BasicInfoHandler report."""
        report = diagnostic_report or {}
        default_params: DefaultParams = {}
        
        for name, strategy_cls in cls._registry.items():
            # 使用 Singleton 模式取得實例
            if strategy_cls not in cls._instances:
                cls._instances[strategy_cls] = strategy_cls()
            
            strategy = cls._instances[strategy_cls]
            default_params[name] = strategy(report)
            
        return default_params

    @classmethod
    def Produce_defaultParameter(
        cls, diagnostic_report: DiagnosticReport | None = None
    ) -> DefaultParams:
        """Backward-compatible alias for older call sites."""
        return cls.produce_default_parameters(diagnostic_report)


# --- 實作策略 ---

@ParameterHandler.register_strategy("misvalue")
class MisvalueStrategy(BaseParameterStrategy):
    # 配置化參數
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


@ParameterHandler.register_strategy("encode")
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


@ParameterHandler.register_strategy("normalize")
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

__all__ = ["ParameterHandler", "BasicInfoHandler"]
