from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from signaltwice.core.types import VisualizeConfig
from signaltwice.engine.plotter import SmartPlotter
from signaltwice.engine.registry import (
    basic_info,
    parameters,
    readers,
    reports,
    visualizers,
    writers,
)

_LOADED_MODULES: set[str] = set()


def ensure_default_strategies(modules: list[str] | None = None) -> None:
    if modules is None:
        modules = [
            "basic_info",
            "parameter",
            "reader",
            "report",
            "visualizer",
            "writer",
        ]

    for module in modules:
        if module in _LOADED_MODULES:
            continue
        if module == "basic_info":
            from signaltwice.strategy import basic_info as _basic_info

            _ = _basic_info
        elif module == "parameter":
            from signaltwice.strategy import parameter as _parameter

            _ = _parameter
        elif module == "reader":
            from signaltwice.strategy import reader as _reader

            _ = _reader
        elif module == "report":
            from signaltwice.strategy import report as _report

            _ = _report
        elif module == "visualizer":
            from signaltwice.strategy import visualizer as _visualizer

            _ = _visualizer
        elif module == "writer":
            from signaltwice.strategy import writer as _writer

            _ = _writer
        else:
            raise ValueError(f"Unknown strategy module: {module}")
        _LOADED_MODULES.add(module)


@dataclass(frozen=True)
class DefaultServices:
    reader: Any
    writer: Any
    parameters: Any
    basic_info: Any
    reporter: Any
    visualizer: Any
    visualizer_registry: Any


def build_default_services(visual_config: VisualizeConfig) -> DefaultServices:
    ensure_default_strategies()
    plotter = SmartPlotter(visual_config, visualizers)
    return DefaultServices(
        reader=readers,
        writer=writers,
        parameters=parameters,
        basic_info=basic_info,
        reporter=reports,
        visualizer=plotter,
        visualizer_registry=visualizers,
    )
