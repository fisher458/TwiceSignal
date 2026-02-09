from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import polars as pl

from signaltwice.core.interface import (
    BaseParameterStrategy,
    BaseReader,
    BaseReportSection,
    BaseVisualizer,
    BaseWriter,
)
from signaltwice.core.types import (
    BasicInfoMetrics,
    DefaultParams,
    DiagnosticReport,
    FrameLike,
    StrategyFn,
    VisualizeConfig,
)
from signaltwice.core.utils import (
    assign_nested,
    build_basic_info_context,
    ensure_lazy,
    is_snake_case,
    ReportStyler,
)


class ReaderRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseReader]] = {}
        self._instances: dict[type[BaseReader], BaseReader] = {}

    def register(self, *extensions: str):
        def wrapper(reader_cls: type[BaseReader]):
            if not issubclass(reader_cls, BaseReader):
                raise TypeError(f"{reader_cls.__name__} must inherit from BaseReader")
            for ext in extensions:
                clean_ext = ext.lstrip(".").lower()
                self._registry[clean_ext] = reader_cls
            return reader_cls

        return wrapper

    def get_strategy(self, name: str) -> BaseReader:
        key = name.lower()
        if key not in self._registry:
            supported = ", ".join(sorted(self._registry.keys()))
            raise ValueError(
                f"Strategy for '{name}' is not registered. Supported: {supported}"
            )

        reader_cls = self._registry[key]
        if reader_cls not in self._instances:
            self._instances[reader_cls] = reader_cls()
        return self._instances[reader_cls]

    def auto_read(
        self,
        file_path: str | Path,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> FrameLike:
        if isinstance(file_path, Path):
            path = file_path
        elif isinstance(file_path, str):
            path = Path(file_path)
        else:
            raise TypeError(
                f"file_path must be str or Path, got {type(file_path).__name__}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path.absolute()}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path.absolute()}")

        if not path.suffix:
            raise ValueError(
                f"File has no extension, cannot infer reader: {path.name}"
            )

        ext = path.suffix.lstrip(".").lower()
        reader = self.get_strategy(ext)
        return reader(path, lazy=lazy, **kwargs)


class WriterRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseWriter]] = {}
        self._instances: dict[type[BaseWriter], BaseWriter] = {}

    def register(self, *formats: str):
        def wrapper(writer_cls: type[BaseWriter]):
            if not issubclass(writer_cls, BaseWriter):
                raise TypeError(f"{writer_cls.__name__} must inherit from BaseWriter")
            for fmt in formats:
                self._registry[fmt.lower().lstrip(".")] = writer_cls
            return writer_cls

        return wrapper

    def get_strategy(self, fmt: str) -> BaseWriter:
        key = fmt.lower().lstrip(".")
        if key not in self._registry:
            supported = ", ".join(sorted(self._registry.keys()))
            raise ValueError(
                f"No writer strategy registered for format '{fmt}'. Supported: {supported}"
            )

        writer_cls = self._registry[key]
        if writer_cls not in self._instances:
            self._instances[writer_cls] = writer_cls()
        return self._instances[writer_cls]

    def auto_write(self, content: Any, file_path: str | Path, **kwargs: Any) -> None:
        path = Path(file_path)
        if not path.suffix:
            raise ValueError(f"File has no extension, cannot infer writer: {path}")

        ext = path.suffix.lstrip(".").lower()
        writer = self.get_strategy(ext)
        writer(content, path, **kwargs)


class ParameterRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[BaseParameterStrategy]] = {}
        self._instances: dict[type[BaseParameterStrategy], BaseParameterStrategy] = {}

    def register(self, name: str):
        def wrapper(strategy_cls: type[BaseParameterStrategy]):
            if not issubclass(strategy_cls, BaseParameterStrategy):
                raise TypeError(
                    f"{strategy_cls.__name__} must inherit from BaseParameterStrategy"
                )
            if name in self._registry:
                existing_cls = self._registry[name].__name__
                raise ValueError(
                    f"Strategy '{name}' is already registered by '{existing_cls}'."
                )
            self._registry[name] = strategy_cls
            return strategy_cls

        return wrapper

    def register_strategy(self, name: str):
        return self.register(name)

    def produce_default_parameters(
        self, diagnostic_report: DiagnosticReport | None = None
    ) -> DefaultParams:
        report = diagnostic_report or {}
        default_params: DefaultParams = {}

        for name, strategy_cls in self._registry.items():
            if strategy_cls not in self._instances:
                self._instances[strategy_cls] = strategy_cls()
            strategy = self._instances[strategy_cls]
            default_params[name] = strategy(report)

        return default_params

    def Produce_defaultParameter(
        self, diagnostic_report: DiagnosticReport | None = None
    ) -> DefaultParams:
        return self.produce_default_parameters(diagnostic_report)


class BasicInfoRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, StrategyFn] = {}

    def register(self, name: str) -> Callable[[StrategyFn], StrategyFn]:
        if not is_snake_case(name):
            raise ValueError(f"Strategy name '{name}' must be snake_case.")

        def wrapper(func: StrategyFn) -> StrategyFn:
            if not is_snake_case(func.__name__):
                raise ValueError(
                    f"Strategy function '{func.__name__}' must be snake_case."
                )
            self._strategies[name] = func
            return func

        return wrapper

    def get_strategy(self, name: str) -> StrategyFn:
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return self._strategies[name]

    def execute_all(self, frame: FrameLike) -> BasicInfoMetrics:
        context = build_basic_info_context(frame)
        outputs: BasicInfoMetrics = {}
        exprs: list[pl.Expr] = []
        strategy_keys: dict[str, list[str]] = {}

        for name, strategy in self._strategies.items():
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

        result = ensure_lazy(frame).select(exprs).collect()
        row = result.row(0, named=True)

        for name, keys in strategy_keys.items():
            strategy_output = outputs[name]
            for key in keys:
                alias = f"{name}___{key}"
                assign_nested(strategy_output, key, row.get(alias))
        return outputs


class ReportRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, type[BaseReportSection]] = {}

    def register_strategy(self, name: str):
        def wrapper(strategy_cls: type[BaseReportSection]):
            self._strategies[name] = strategy_cls
            return strategy_cls

        return wrapper

    def register(self, name: str):
        return self.register_strategy(name)

    def generate_full_report(
        self,
        raw_data: dict,
        execution_order: list[str] | None = None,
        enable_color: bool = True,
    ) -> str:
        report_parts: list[str] = []
        styler = ReportStyler(enable_color=enable_color)
        order = [
            "dataOverview",
            "missingValueReport",
            "numericStats",
            "categoricalStats",
            "dataIntegrity",
        ]
        base_order = execution_order or order
        keys_to_run = [k for k in base_order if k in self._strategies]
        extra_keys = [k for k in self._strategies.keys() if k not in keys_to_run]
        keys_to_run.extend(extra_keys)

        for name in keys_to_run:
            handler = self._strategies[name]()
            metrics = handler.calculate(raw_data)
            section_text = handler.render(metrics, styler)
            report_parts.append(section_text)

        report_parts.extend(styler.separator())
        return "\n".join(report_parts)


class VisualizerRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, type[BaseVisualizer]] = {}

    def register(self, name: str):
        def wrapper(visualizer_cls: type[BaseVisualizer]):
            self._strategies[name] = visualizer_cls
            return visualizer_cls

        return wrapper

    def register_visualize_strategy(self, name: str):
        return self.register(name)

    def create_visualizer(self, name: str, config: VisualizeConfig) -> BaseVisualizer:
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return self._strategies[name](config)


readers = ReaderRegistry()
writers = WriterRegistry()
parameters = ParameterRegistry()
basic_info = BasicInfoRegistry()
reports = ReportRegistry()
visualizers = VisualizerRegistry()

__all__ = [
    "ReaderRegistry",
    "WriterRegistry",
    "ParameterRegistry",
    "BasicInfoRegistry",
    "ReportRegistry",
    "VisualizerRegistry",
    "readers",
    "writers",
    "parameters",
    "basic_info",
    "reports",
    "visualizers",
]
