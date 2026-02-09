"""
#到時候需要override這些參數

from signaltwice.diagnose import DiagnosticPipeline, OutputConfig

pipeline = DiagnosticPipeline(
    output_dir="data/report",
    plot_format="pdf",
    reader_lazy=True,
)

result = pipeline.run_diagnosis(
    "data/sample.csv",
    lazy=None,  # None -> 使用 reader_lazy
    output_config=OutputConfig(
        report_path="data/report/diagnostic_report.txt",
        plot_dir="data/report/plots",
        parameter_path="data/report/suggested_params.json",
    ),
    plot_dpi=300,
    image_quality=85,
)
"""

from __future__ import annotations

import sys
import logging as py_logging
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json

import polars as pl

from signaltwice.core.protocol import (
    BasicInfoService,
    ParameterService,
    ReaderService,
    ReportService,
    VisualizerService,
    WriterService,
)
from signaltwice.core.types import OutputConfig, VisualizeConfig
from signaltwice.engine.defaults import build_default_services
from signaltwice.logging import get_logger, configure_logging


class DiagnosticPipeline:
    """
    Orchestrates the diagnostic workflow by wiring together:
    reader -> getBasicInfo -> visualizer/reportSection -> writer.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        plot_format: str = "pdf",
        reader_lazy: bool = True,
        reader: ReaderService | None = None,
        writer: WriterService | None = None,
        parameters: ParameterService | None = None,
        basic_info: BasicInfoService | None = None,
        reporter: ReportService | None = None,
        visualizer: VisualizerService | None = None,
        logger: py_logging.Logger | None = None,
    ) -> None:
        # Output settings.
        self.default_output_dir = (
            Path(output_dir) if output_dir else Path.cwd() / "data" / "report"
        )
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        self.reader_lazy = reader_lazy

        # Visualizer configuration (plots are saved through writer below).
        self.visual_config = VisualizeConfig(
            format=plot_format,
            output_dir=self.default_output_dir,
        )

        defaults = None
        if any(
            dep is None
            for dep in (reader, writer, parameters, basic_info, reporter, visualizer)
        ):
            defaults = build_default_services(self.visual_config)

        if defaults is None:
            self.reader = reader
            self.writer = writer
            self.parameter_handler = parameters
            self.basic_info = basic_info
            self.reporter = reporter
            self.plotter = visualizer
        else:
            self.reader = reader if reader is not None else defaults.reader
            self.writer = writer if writer is not None else defaults.writer
            self.parameter_handler = (
                parameters if parameters is not None else defaults.parameters
            )
            self.basic_info = basic_info if basic_info is not None else defaults.basic_info
            self.reporter = reporter if reporter is not None else defaults.reporter
            self.plotter = visualizer if visualizer is not None else defaults.visualizer

        self.logger = logger or get_logger(__name__)

    def run_diagnosis(
        self,
        file_path: str | Path,
        *,
        lazy: bool | None = None,
        output_config: OutputConfig | None = None,
        plot_dpi: int | None = None,
        image_quality: int | None = None,
    ) -> Dict[str, Any]:
        """
        Execute the end-to-end diagnosis pipeline.
        Returns a dict with report, plots, metrics, and parameters.
        """
        result: Dict[str, Any] = {
            "file_path": str(file_path),
            "status": {},
            "errors": [],
        }

        def mark_step(step: str, ok: bool, error: str | None = None) -> None:
            entry: Dict[str, Any] = {"ok": ok}
            if error:
                entry["error"] = error
                result["errors"].append(error)
            result["status"][step] = entry

        def finalize_result(required_steps: Iterable[str], optional_steps: Iterable[str]) -> Dict[str, Any]:
            success = all(
                result["status"].get(step, {}).get("ok") for step in required_steps
            )
            result["success"] = success
            result["partial_success"] = success and any(
                step in result["status"] and not result["status"][step]["ok"]
                for step in optional_steps
            )
            if not success and "error" not in result and result["errors"]:
                result["error"] = result["errors"][-1]
            return result

        required_steps = ("extract", "preprocess", "diagnostics")
        optional_steps = ("parameters", "report", "plots")
        self.logger.info("Step start: Extract (read data) | file_path=%s", file_path)

        # 1) Extract: load dataset with reader.
        try:
            frame = self.reader.auto_read(
                file_path, lazy=lazy if lazy is not None else self.reader_lazy
            )
            self.logger.info("Step end: Extract (read data)")
            mark_step("extract", True)
        except Exception as exc:
            self.logger.exception("Extract failed: read input file | file_path=%s", file_path)
            error_msg = f"Extract failed: read input file: {exc}"
            mark_step("extract", False, error_msg)
            for step in required_steps:
                if step != "extract" and step not in result["status"]:
                    mark_step(step, False, "Skipped due to extract failure.")
            for step in optional_steps:
                if step not in result["status"]:
                    mark_step(step, False, "Skipped due to extract failure.")
            result["error"] = error_msg
            return finalize_result(required_steps, optional_steps)

        # 2) Transform: clean/preprocess rows and columns.
        self.logger.info("Step start: Transform (preprocess data)")
        try:
            cleaned = self._preprocess(frame)
            if not isinstance(cleaned, (pl.DataFrame, pl.LazyFrame)):
                raise TypeError("Preprocess must return a polars DataFrame or LazyFrame.")
            self.logger.info("Step end: Transform (preprocess data)")
            mark_step("preprocess", True)
        except Exception as exc:  # pragma: no cover - defensive safety net
            self.logger.exception("Transform failed: preprocess data")
            error_msg = f"Transform failed: preprocess data: {exc}"
            mark_step("preprocess", False, error_msg)
            for step in required_steps:
                if step not in result["status"]:
                    mark_step(step, False, "Skipped due to preprocess failure.")
            for step in optional_steps:
                if step not in result["status"]:
                    mark_step(step, False, "Skipped due to preprocess failure.")
            result["error"] = error_msg
            return finalize_result(required_steps, optional_steps)

        # 3) Transform: compute diagnostic metrics.
        self.logger.info("Step start: Transform (compute diagnostics)")
        try:
            metrics = self.basic_info.execute_all(cleaned)
            if not isinstance(metrics, dict):
                raise ValueError("Diagnostics must return a dictionary of metrics.")
            self.logger.info("Step end: Transform (compute diagnostics)")
            mark_step("diagnostics", True)
        except Exception as exc:  # pragma: no cover - defensive safety net
            self.logger.exception("Transform failed: compute diagnostics")
            error_msg = f"Transform failed: compute diagnostics: {exc}"
            mark_step("diagnostics", False, error_msg)
            for step in optional_steps:
                if step not in result["status"]:
                    mark_step(step, False, "Skipped due to diagnostics failure.")
            result["error"] = error_msg
            return finalize_result(required_steps, optional_steps)

        # 4) Configure: derive default parameters from diagnostic metrics.
        self.logger.info("Step start: Configure (derive default parameters)")
        default_params: Dict[str, Any] | None = None
        parameter_path: Path | None = None
        try:
            default_params = self.parameter_handler.produce_default_parameters(metrics)
            self.logger.info("Step end: Configure (derive default parameters)")

            # 4.1) Persist parameters (JSON).
            self.logger.info("Step start: Report (write parameters)")
            parameter_path = self._write_parameters(default_params, output_config)
            if not parameter_path.exists():
                raise FileNotFoundError(f"Parameter file not written: {parameter_path}")
            self.logger.info("Step end: Report (write parameters)")
            mark_step("parameters", True)
        except Exception as exc:
            self.logger.exception("Configure/Report failed: parameters")
            error_msg = f"Configure/Report failed: parameters: {exc}"
            mark_step("parameters", False, error_msg)
            parameter_path = None

        # 5) Report: format metrics into a structured text report.
        self.logger.info("Step start: Report (write report)")
        report_text: str | None = None
        report_path: Path | None = None
        try:
            report_text = self.reporter.generate_full_report(metrics)
            report_path = self._write_report(report_text, output_config)
            if not report_path.exists():
                raise FileNotFoundError(f"Report file not written: {report_path}")
            self.logger.info("Step end: Report (write report)")
            mark_step("report", True)
        except Exception as exc:
            self.logger.exception("Report failed: write report")
            error_msg = f"Report failed: write report: {exc}"
            mark_step("report", False, error_msg)
            report_path = None

        # 6) Visualize: generate plots and persist them.
        self.logger.info("Step start: Visualize (write plots)")
        plot_paths: List[Path] = []
        try:
            plot_frame = cleaned
            if isinstance(plot_frame, pl.LazyFrame):
                plot_frame = plot_frame.collect()
            if not isinstance(plot_frame, pl.DataFrame):
                raise TypeError("Visualizer requires a polars DataFrame.")
            plot_paths = self._write_plots(
                plot_frame,
                metrics,
                output_config,
                dpi=plot_dpi,
                image_quality=image_quality,
            )
            self.logger.info("Step end: Visualize (write plots)")
            mark_step("plots", True)
        except Exception as exc:
            self.logger.exception("Visualize failed: write plots")
            error_msg = f"Visualize failed: write plots: {exc}"
            mark_step("plots", False, error_msg)

        result.update(
            {
                "metrics": metrics,
                "default_parameters": default_params,
                "report_path": str(report_path) if report_path else None,
                "parameter_path": str(parameter_path) if parameter_path else None,
                "plot_paths": [str(path) for path in plot_paths],
            }
        )
        return finalize_result(required_steps, optional_steps)

    def _preprocess(
        self, frame: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame | pl.LazyFrame:
        if not isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError("Reader must return a polars DataFrame or LazyFrame.")

        if isinstance(frame, pl.LazyFrame):
            return self._preprocess_lazy(frame)

        return self._preprocess_eager(frame)

    def _preprocess_eager(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.height == 0 or frame.width == 0:
            self.logger.debug(
                "Empty frame detected | rows=%s cols=%s",
                frame.height,
                frame.width,
            )
            return frame

        self.logger.debug("Raw frame shape | rows=%s cols=%s", frame.height, frame.width)
        self.logger.debug("Raw columns | columns=%s", frame.columns)

        # Drop columns that are entirely null.
        all_null_cols = [
            col for col in frame.columns if frame[col].null_count() == frame.height
        ]
        if all_null_cols:
            self.logger.debug("Dropping all-null columns | columns=%s", all_null_cols)
            frame = frame.drop(all_null_cols)

        # Drop rows that are entirely null (handles specific rows).
        if frame.width:
            null_exprs = [pl.col(col).is_null() for col in frame.columns]
            non_empty_rows = pl.sum_horizontal(null_exprs) < frame.width
            before_rows = frame.height
            frame = frame.filter(non_empty_rows)
            if frame.height != before_rows:
                self.logger.debug(
                    "Filtered empty rows | before_rows=%s after_rows=%s",
                    before_rows,
                    frame.height,
                )

        self.logger.debug("Post-preprocess shape | rows=%s cols=%s", frame.height, frame.width)
        return frame

    def _preprocess_lazy(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        columns = list(frame.collect_schema().keys())
        if not columns:
            self.logger.debug("Empty schema detected | cols=0")
            return frame

        stats_exprs = [pl.len().alias("__row_count")]
        stats_exprs.extend([pl.col(col).null_count().alias(col) for col in columns])
        stats = frame.select(stats_exprs).collect()
        row = stats.row(0, named=True)
        row_count = row.pop("__row_count", 0)
        if row_count == 0:
            self.logger.debug("Empty frame detected | rows=0 cols=%s", len(columns))
            return frame

        all_null_cols = [col for col, count in row.items() if count == row_count]
        if all_null_cols:
            self.logger.debug("Dropping all-null columns | columns=%s", all_null_cols)
            frame = frame.drop(all_null_cols)

        columns = list(frame.collect_schema().keys())
        if columns:
            null_exprs = [pl.col(col).is_null() for col in columns]
            non_empty_rows = pl.sum_horizontal(null_exprs) < len(columns)
            frame = frame.filter(non_empty_rows)

        self.logger.debug("Post-preprocess lazy schema | cols=%s", len(columns))
        return frame

    def _write_report(self, report_text: str, config: OutputConfig | None) -> Path:
        report_path = None
        if config and config.report_path:
            report_path = Path(config.report_path)
        if report_path is None:
            report_path = self.default_output_dir / "diagnostic_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = self.writer.get_strategy("text")
            writer_instance = writer() if isinstance(writer, type) else writer
            writer_instance(report_text, str(report_path))
        except Exception:
            self.logger.exception(
                "Report failed: write report | report_path=%s", report_path
            )
            # Fallback to direct write if writer strategy is not ready.
            report_path.write_text(report_text, encoding="utf-8")

        if not report_path.exists():
            report_path.write_text(report_text, encoding="utf-8")
        return report_path

    def _write_parameters(
        self,
        params: Dict[str, Any],
        config: OutputConfig | None,
    ) -> Path:
        parameter_path = None
        if config and config.parameter_path:
            parameter_path = Path(config.parameter_path)
        if parameter_path is None:
            parameter_path = self.default_output_dir / "suggested_params.json"
        parameter_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            parameter_path.write_text(
                json.dumps(params, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            self.logger.exception(
                "Report failed: write parameters | parameter_path=%s", parameter_path
            )
            # Last-resort fallback: attempt a minimal JSON write.
            parameter_path.write_text("{}", encoding="utf-8")
        return parameter_path

    def _write_plots(
        self,
        frame: pl.DataFrame,
        metrics: Dict[str, Dict[str, Any]],
        config: OutputConfig | None,
        *,
        dpi: int | None = None,
        image_quality: int | None = None,
    ) -> List[Path]:
        plot_paths: List[Path] = []
        plot_dir = None
        if config and config.plot_dir:
            plot_dir = Path(config.plot_dir)
        if plot_dir is None:
            plot_dir = self.default_output_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

        original_dpi = self.visual_config.dpi
        if dpi is not None:
            self.visual_config.dpi = dpi

        def resolve_save_kwargs() -> Dict[str, Any]:
            save_kwargs: Dict[str, Any] = {"dpi": self.visual_config.dpi}
            if image_quality is None:
                return save_kwargs
            fmt = self.visual_config.format.lower()
            if fmt in {"jpg", "jpeg", "webp"}:
                quality = max(1, min(95, int(image_quality)))
                save_kwargs["quality"] = quality
            elif fmt == "png":
                quality = max(1, min(95, int(image_quality)))
                compress_level = max(0, min(9, 9 - round((quality - 1) / 94 * 9)))
                save_kwargs["compress_level"] = compress_level
            return save_kwargs

        save_kwargs = resolve_save_kwargs()

        def safe_dir_name(value: str) -> str:
            return "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in value
            )

        def save_plot(label: str, strategy: str, fig: Any) -> None:
            metric_dir = plot_dir / safe_dir_name(strategy)
            metric_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{label}.{self.visual_config.format}"
            plot_path = metric_dir / filename

            try:
                writer = self.writer.get_strategy(self.visual_config.format)
                writer_instance = writer() if isinstance(writer, type) else writer
                writer_instance(fig, str(plot_path), **save_kwargs)
            except Exception:
                self.logger.exception(
                    "Visualize failed: write plot | plot_path=%s", plot_path
                )
                # Fallback to direct save if writer strategy is not ready.
                fig.savefig(
                    plot_path, format=self.visual_config.format, **save_kwargs
                )

            if not plot_path.exists():
                fig.savefig(
                    plot_path, format=self.visual_config.format, **save_kwargs
                )

            plot_paths.append(plot_path)

        try:
            self.plotter.run_batch(frame, metrics, save_plot)
        finally:
            if dpi is not None:
                self.visual_config.dpi = original_dpi
        return plot_paths

    def _fail(self, message: str, exc: Exception, result: Dict[str, Any]) -> Dict[str, Any]:
        error_msg = f"{message}: {exc}"
        self.logger.error(error_msg)
        print(error_msg, file=sys.stderr)
        result["error"] = error_msg
        return result


if __name__ == "__main__":
    configure_logging()
    if len(sys.argv) < 2:
        print("Usage: python -m signaltwice.diagnose <file_path>", file=sys.stderr)
        raise SystemExit(1)

    pipeline = DiagnosticPipeline()
    pipeline.run_diagnosis(sys.argv[1])
