from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from signaltwice.core.interface import BaseReader
from signaltwice.engine.registry import readers


@readers.register("csv", "txt")
class CsvReader(BaseReader):
    def __call__(
        self,
        file_path: str | Path,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> pl.DataFrame | pl.LazyFrame:
        if lazy:
            return pl.scan_csv(file_path, **kwargs)
        return pl.read_csv(file_path, **kwargs)


@readers.register("parquet", "pq")
class ParquetReader(BaseReader):
    def __call__(
        self,
        file_path: str | Path,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> pl.DataFrame | pl.LazyFrame:
        if lazy:
            return pl.scan_parquet(file_path, **kwargs)
        return pl.read_parquet(file_path, **kwargs)
