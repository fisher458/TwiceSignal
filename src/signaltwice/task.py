from __future__ import annotations

import sys
from types import ModuleType
from typing import Callable


__path__ = []  # Mark as namespace-like package for submodule imports.


def _lazy_module(name: str, loader: Callable[[ModuleType], None]) -> ModuleType:
    module = ModuleType(name)
    module.__dict__["_loaded"] = False

    def _load() -> None:
        if module.__dict__.get("_loaded"):
            return
        loader(module)
        module.__dict__["_loaded"] = True

    def __getattr__(attr: str):  # type: ignore[override]
        _load()
        return module.__dict__[attr]

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _load_reader(module: ModuleType) -> None:
    from signaltwice.engine.defaults import ensure_default_strategies
    from signaltwice.engine.registry import readers as ReadHandler
    from signaltwice.core.interface import BaseReader

    ensure_default_strategies(["reader"])
    module.ReadHandler = ReadHandler
    module.BaseReader = BaseReader
    module.__all__ = ["ReadHandler", "BaseReader"]


def _load_writer(module: ModuleType) -> None:
    from signaltwice.engine.defaults import ensure_default_strategies
    from signaltwice.engine.registry import writers as WriteHandler
    from signaltwice.core.interface import BaseWriter

    ensure_default_strategies(["writer"])
    module.WriteHandler = WriteHandler
    module.BaseWriter = BaseWriter
    module.__all__ = ["WriteHandler", "BaseWriter"]


def _load_default_parameter(module: ModuleType) -> None:
    from signaltwice.engine.defaults import ensure_default_strategies
    from signaltwice.engine.registry import basic_info as BasicInfoHandler
    from signaltwice.engine.registry import parameters as ParameterHandler
    from signaltwice.core.interface import BaseParameterStrategy
    from signaltwice.core.types import DiagnosticReport, DefaultParams

    ensure_default_strategies(["parameter", "basic_info"])
    module.ParameterHandler = ParameterHandler
    module.BasicInfoHandler = BasicInfoHandler
    module.BaseParameterStrategy = BaseParameterStrategy
    module.DiagnosticReport = DiagnosticReport
    module.DefaultParams = DefaultParams
    module.__all__ = [
        "ParameterHandler",
        "BasicInfoHandler",
        "BaseParameterStrategy",
        "DiagnosticReport",
        "DefaultParams",
    ]


def _load_basic_info(module: ModuleType) -> None:
    from signaltwice.engine.defaults import ensure_default_strategies
    from signaltwice.engine.registry import basic_info as BasicInfoHandler
    from signaltwice.core.types import StrategyContext, StrategyExprs, StrategyFn

    ensure_default_strategies(["basic_info"])
    module.BasicInfoHandler = BasicInfoHandler
    module.StrategyContext = StrategyContext
    module.StrategyExprs = StrategyExprs
    module.StrategyFn = StrategyFn
    module.__all__ = ["BasicInfoHandler", "StrategyContext", "StrategyExprs", "StrategyFn"]


def _load_report_section(module: ModuleType) -> None:
    from signaltwice.engine.defaults import ensure_default_strategies
    from signaltwice.engine.registry import reports as ReportSectionHandler
    from signaltwice.core.utils import ReportStyler, SafeExtractor

    ensure_default_strategies(["report"])
    module.ReportSectionHandler = ReportSectionHandler
    module.ReportStyler = ReportStyler
    module.SafeExtractor = SafeExtractor
    module.__all__ = ["ReportSectionHandler", "ReportStyler", "SafeExtractor"]


def _load_visualizer(module: ModuleType) -> None:
    from signaltwice.engine.defaults import ensure_default_strategies
    from signaltwice.engine.registry import visualizers as VisualizeHandler
    from signaltwice.engine.plotter import SmartPlotter
    from signaltwice.core.interface import BaseVisualizer
    from signaltwice.core.types import VisualizeConfig

    ensure_default_strategies(["visualizer"])
    module.BaseVisualizer = BaseVisualizer
    module.VisualizeConfig = VisualizeConfig
    module.SmartPlotter = SmartPlotter
    module.VisualizeHandler = VisualizeHandler
    module.__all__ = [
        "BaseVisualizer",
        "VisualizeConfig",
        "SmartPlotter",
        "VisualizeHandler",
    ]


reader = _lazy_module(f"{__name__}.reader", _load_reader)
writer = _lazy_module(f"{__name__}.writer", _load_writer)
defaultParameter = _lazy_module(f"{__name__}.defaultParameter", _load_default_parameter)
getBasicInfo = _lazy_module(f"{__name__}.getBasicInfo", _load_basic_info)
reportSection = _lazy_module(f"{__name__}.reportSection", _load_report_section)
visualizer = _lazy_module(f"{__name__}.visualizer", _load_visualizer)

__all__ = [
    "reader",
    "writer",
    "defaultParameter",
    "getBasicInfo",
    "reportSection",
    "visualizer",
]
