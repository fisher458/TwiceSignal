from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

# 1. 改進：引入 Template Method Pattern (樣板方法模式)
class BaseWriter(ABC):
    def __call__(self, content: Any, file_path: str | Path, **kwargs: Any) -> None:
        """
        對外統一的介面：負責處理共用邏輯 (如建立目錄)。
        不要 override 這個方法，請 override _write。
        """
        path = Path(file_path)
        
        # DRY: 統一在這裡處理目錄建立
        # 改進：避免對當前目錄 ('.') 呼叫 mkdir，雖然不報錯但省一個 syscall
        if path.parent and path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # 呼叫實際的寫入邏輯
        self._write(content, path, **kwargs)

    @abstractmethod
    def _write(self, content: Any, path: Path, **kwargs: Any) -> None:
        """
        [Hook Method] 子類別必須實作此方法來執行真正的寫入。
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Instance>"


class WriteHandler:
    _registry: ClassVar[dict[str, type[BaseWriter]]] = {}
    _instances: ClassVar[dict[type[BaseWriter], BaseWriter]] = {}

    @classmethod
    def register(cls, *formats: str):
        def wrapper(writer_cls: type[BaseWriter]):
            if not issubclass(writer_cls, BaseWriter):
                raise TypeError(f"{writer_cls.__name__} must inherit from BaseWriter")
            for fmt in formats:
                cls._registry[fmt.lower().lstrip(".")] = writer_cls
            return writer_cls
        return wrapper

    @classmethod
    def get_strategy(cls, fmt: str) -> BaseWriter:
        fmt = fmt.lower().lstrip(".")
        if fmt not in cls._registry:
            supported = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"No writer strategy registered for format '{fmt}'. Supported: {supported}")
        
        writer_cls = cls._registry[fmt]
        if writer_cls not in cls._instances:
            cls._instances[writer_cls] = writer_cls()
            
        return cls._instances[writer_cls]

    @classmethod
    def auto_write(cls, content: Any, file_path: str | Path, **kwargs: Any) -> None:
        path = Path(file_path)
        
        if not path.suffix:
             raise ValueError(f"File has no extension, cannot infer writer: {path}")
             
        ext = path.suffix.lstrip('.').lower()
        
        try:
            writer = cls.get_strategy(ext)
            writer(content, path, **kwargs)
        except ValueError as e:
            # 這裡可以捕捉並拋出更具體的錯誤，例如 PermissionError
            raise e


# --- 實作策略 ---

@WriteHandler.register("text", "txt")
class TextWriter(BaseWriter):
    def _write(self, content: Any, path: Path, **kwargs: Any) -> None:
        if not isinstance(content, str):
            content = str(content)
        encoding = kwargs.get("encoding", "utf-8")
        path.write_text(content, encoding=encoding)


@WriteHandler.register("csv")
class CsvWriter(BaseWriter):
    """
    支援 Polars DataFrame, Pandas DataFrame 或 List[Dict] 寫入 CSV。
    """
    def _write(self, content: Any, path: Path, **kwargs: Any) -> None:
        # 優先支援 Polars
        if hasattr(content, "write_csv"):
            content.write_csv(path, **kwargs)
            return
        
        # 支援 Pandas
        if hasattr(content, "to_csv"):
            # Pandas 的 index 預設是 True，通常存資料時不需要
            if "index" not in kwargs:
                kwargs["index"] = False
            content.to_csv(path, **kwargs)
            return

        raise TypeError(f"Content type {type(content)} does not support direct CSV writing.")


@WriteHandler.register("png", "pdf", "svg", "jpg", "jpeg")
class PlotWriter(BaseWriter):
    def _write(self, fig: Any, path: Path, **kwargs: Any) -> None:
        # Matplotlib
        if hasattr(fig, "savefig"):
            fig.savefig(path, **kwargs)
            # 注意：這裡不主動 plt.close(fig)，因為 fig 生命週期應由呼叫者管理
            return

        # Plotly
        if hasattr(fig, "write_image"):
            fig.write_image(str(path), **kwargs)
            return

        raise TypeError(f"Unsupported figure object: {type(fig)}")


@WriteHandler.register("yaml", "yml")
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

# --- 測試 ---
if __name__ == "__main__":
    try:
        # 測試 CSV 寫入 (模擬 Polars)
        import polars as pl
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        WriteHandler.auto_write(df, "output/data.csv")
        print("CSV written successfully.")
        
        # 測試單例
        print(f"Singleton Check: {WriteHandler.get_strategy('csv') is WriteHandler.get_strategy('csv')}")

    except Exception as e:
        print(f"Error: {e}")