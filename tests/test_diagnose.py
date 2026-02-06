from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

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
