from __future__ import annotations

from unittest.mock import Mock

import polars as pl

from signaltwice.diagnose import DiagnosticPipeline, OutputConfig


def _make_pipeline(tmp_path) -> DiagnosticPipeline:
    pipeline = DiagnosticPipeline(output_dir=tmp_path)
    pipeline.reader = Mock()
    pipeline.writer = Mock()
    return pipeline


def test_preprocess_lazy_drops_all_nulls(tmp_path) -> None:
    pipeline = _make_pipeline(tmp_path)
    frame = pl.DataFrame(
        {
            "a": [1, None, None],
            "b": [None, None, None],
            "c": [None, 2, None],
        }
    )

    cleaned = pipeline._preprocess(frame.lazy())

    assert isinstance(cleaned, pl.LazyFrame)
    result = cleaned.collect()
    assert result.columns == ["a", "c"]
    assert result.height == 2


def test_preprocess_eager_drops_all_nulls(tmp_path) -> None:
    pipeline = _make_pipeline(tmp_path)
    frame = pl.DataFrame(
        {
            "a": [1, None, None],
            "b": [None, None, None],
            "c": [None, 2, None],
        }
    )

    cleaned = pipeline._preprocess(frame)

    assert isinstance(cleaned, pl.DataFrame)
    assert cleaned.columns == ["a", "c"]
    assert cleaned.height == 2


def test_write_report_uses_writer_and_writes_file(tmp_path) -> None:
    pipeline = _make_pipeline(tmp_path)
    writer_instance = Mock()
    pipeline.writer.get_strategy.return_value = writer_instance

    report_text = "Diagnostic report"
    config = OutputConfig(report_path=tmp_path / "diagnostic_report.txt")

    report_path = pipeline._write_report(report_text, config)

    pipeline.writer.get_strategy.assert_called_once_with("text")
    writer_instance.assert_called_once_with(report_text, str(report_path))
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8") == report_text


class _NoopPlotter:
    def run_batch(self, frame, metrics, save_fn) -> None:  # pragma: no cover - trivial
        return None


def test_run_diagnosis_reader_failure_marks_required_steps(tmp_path) -> None:
    pipeline = DiagnosticPipeline(output_dir=tmp_path)
    pipeline.reader = Mock()
    pipeline.reader.auto_read.side_effect = FileNotFoundError("missing")

    result = pipeline.run_diagnosis("missing.csv")

    assert result["success"] is False
    assert result["status"]["extract"]["ok"] is False
    assert result["status"]["preprocess"]["ok"] is False
    assert result["status"]["diagnostics"]["ok"] is False
    assert result["error"].startswith("Extract failed")


def test_run_diagnosis_report_write_failure_marks_report_failed(tmp_path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\\n1,foo\\n2,bar\\n", encoding="utf-8")

    report_dir = tmp_path / "report_path"
    report_dir.mkdir()

    pipeline = DiagnosticPipeline(output_dir=tmp_path, reader_lazy=False)
    pipeline.plotter = _NoopPlotter()

    result = pipeline.run_diagnosis(
        csv_path,
        output_config=OutputConfig(
            report_path=report_dir,
            plot_dir=tmp_path / "plots",
            parameter_path=tmp_path / "params.json",
        ),
    )

    assert result["success"] is True
    assert result["status"]["report"]["ok"] is False
    assert result["status"]["diagnostics"]["ok"] is True


def test_run_diagnosis_plot_dir_failure_marks_plots_failed(tmp_path) -> None:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("a,b\\n1,foo\\n2,bar\\n", encoding="utf-8")

    plot_file = tmp_path / "plots"
    plot_file.write_text("not a dir", encoding="utf-8")

    pipeline = DiagnosticPipeline(output_dir=tmp_path, reader_lazy=False)
    pipeline.plotter = _NoopPlotter()

    result = pipeline.run_diagnosis(
        csv_path,
        output_config=OutputConfig(
            report_path=tmp_path / "report.txt",
            plot_dir=plot_file,
            parameter_path=tmp_path / "params.json",
        ),
    )

    assert result["success"] is True
    assert result["status"]["plots"]["ok"] is False
    assert result["status"]["diagnostics"]["ok"] is True
