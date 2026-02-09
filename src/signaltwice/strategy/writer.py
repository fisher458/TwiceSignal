from __future__ import annotations

from pathlib import Path
from typing import Any

from signaltwice.core.interface import BaseWriter
from signaltwice.engine.registry import writers


@writers.register("text", "txt")
class TextWriter(BaseWriter):
    def _write(self, content: Any, path: Path, **kwargs: Any) -> None:
        if not isinstance(content, str):
            content = str(content)
        encoding = kwargs.get("encoding", "utf-8")
        path.write_text(content, encoding=encoding)


@writers.register("csv")
class CsvWriter(BaseWriter):
    def _write(self, content: Any, path: Path, **kwargs: Any) -> None:
        if hasattr(content, "write_csv"):
            content.write_csv(path, **kwargs)
            return

        if hasattr(content, "to_csv"):
            if "index" not in kwargs:
                kwargs["index"] = False
            content.to_csv(path, **kwargs)
            return

        raise TypeError(
            f"Content type {type(content)} does not support direct CSV writing."
        )


@writers.register("png", "pdf", "svg", "jpg", "jpeg")
class PlotWriter(BaseWriter):
    def _write(self, fig: Any, path: Path, **kwargs: Any) -> None:
        if hasattr(fig, "savefig"):
            fig.savefig(path, **kwargs)
            return

        if hasattr(fig, "write_image"):
            fig.write_image(str(path), **kwargs)
            return

        raise TypeError(f"Unsupported figure object: {type(fig)}")


@writers.register("yaml", "yml")
class YamlWriter(BaseWriter):
    def _write(self, content: Any, path: Path, **kwargs: Any) -> None:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required.") from exc

        dump_kwargs = {
            "allow_unicode": True,
            "sort_keys": False,
            "default_flow_style": False,
        }
        dump_kwargs.update(kwargs)
        file_encoding = dump_kwargs.pop("encoding", "utf-8")

        with path.open("w", encoding=file_encoding) as handle:
            yaml.safe_dump(content, handle, **dump_kwargs)
