from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Any

import polars as pl

class BaseReader(ABC):
    @abstractmethod
    def __call__(
        self,
        file_path: str | Path,
        *,
        lazy: bool = True,
        **kwargs: Any,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        執行讀取操作。
        
        :param file_path: 檔案路徑
        :param lazy: True 回傳 LazyFrame (scan), False 回傳 DataFrame (read)
        :param kwargs: 透傳給 polars 的參數 (如 separator, has_header 等)
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Instance>"

class ReadHandler:
    _registry: ClassVar[dict[str, type[BaseReader]]] = {}
    # 使用類別作為 Key，確保真正的單例
    _instances: ClassVar[dict[type[BaseReader], BaseReader]] = {}

    @classmethod
    def register(cls, *extensions: str):
        def wrapper(reader_cls: type[BaseReader]):
            if not issubclass(reader_cls, BaseReader):
                raise TypeError(f"{reader_cls.__name__} must inherit from BaseReader")
            
            for ext in extensions:
                clean_ext = ext.lstrip('.').lower()
                cls._registry[clean_ext] = reader_cls
            return reader_cls
        return wrapper

    @classmethod
    def get_strategy(cls, name: str) -> BaseReader:
        name = name.lower()
        if name not in cls._registry:
            # 友善的錯誤提示，列出支援的格式
            supported = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Strategy for '{name}' is not registered. Supported: {supported}")
        
        reader_cls = cls._registry[name]
        
        if reader_cls not in cls._instances:
            cls._instances[reader_cls] = reader_cls()

        return cls._instances[reader_cls]
    
    @classmethod
    def auto_read(cls, file_path: str | Path, lazy: bool = True, **kwargs: Any) -> pl.DataFrame | pl.LazyFrame:
        # 型別防禦與統一
        if isinstance(file_path, Path):
            path = file_path
        elif isinstance(file_path, str):
            path = Path(file_path)
        else:
            raise TypeError(f"file_path must be str or Path, got {type(file_path).__name__}")

        # 存在性檢查
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path.absolute()}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path.absolute()}")

        # 副檔名檢查
        if not path.suffix:
            raise ValueError(f"File has no extension, cannot infer reader: {path.name}")
             
        ext = path.suffix.lstrip('.').lower()
        
        try:
            reader = cls.get_strategy(ext)
            return reader(path, lazy=lazy, **kwargs)
        except ValueError as e:
            # 這裡可以選擇直接拋出，或者包裝更多訊息
            raise e

# --- 實作策略 ---

@ReadHandler.register("csv", "txt")
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

@ReadHandler.register("parquet", "pq")
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

# --- 測試 ---
if __name__ == "__main__":
    try:
        # 測試單例邏輯
        r1 = ReadHandler.get_strategy("csv")
        r2 = ReadHandler.get_strategy("txt")
        print(f"Reader 1: {r1}")
        print(f"Reader 2: {r2}")
        print(f"Is True Singleton? {r1 is r2}") # True
        
        # 測試錯誤處理 (假設檔案不存在)
        # ReadHandler.auto_read("ghost_file.csv")
        
    except Exception as e:
        print(f"Error: {e}")