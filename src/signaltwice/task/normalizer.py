import polars as pl
from abc import ABC, abstractmethod



class NormalizeHandler:
    _normalizeStrategy = {}
    
    @classmethod
    def register_normalize_strategy(cls, name: str):
        def wrapper(strategy_cls):
            cls._normalizeStrategy[name] = strategy_cls
            return strategy_cls
        return wrapper
    
    @classmethod
    def get_strategy(cls, method: str):
        """根據名稱取得策略類別"""
        if method not in cls._normalizeStrategy:
            raise ValueError(f"Strategy '{method}' is not registered.")
        return cls._normalizeStrategy[method]


class baseNormalizer(ABC):
    
    def __init__(self, lf: pl.LazyFrame):
        self.lf = lf

    @abstractmethod
    def __call__(self):
        pass

@NormalizeHandler.register_normalize_strategy("minmax")
class MinMaxNormalizer(baseNormalizer):
    
    def __call__(self):
        pass

@NormalizeHandler.register_normalize_strategy("zscore")
class ZScoreNormalizer(baseNormalizer):
    
    def __call__(self):
        pass

@NormalizeHandler.register_normalize_strategy("robust")
class RobustNormalizer(baseNormalizer):
    
    def __call__(self):
        pass

@NormalizeHandler.register_normalize_strategy("maxabs")
class MaxAbsNormalizer(baseNormalizer):
   
    def __call__(self):
        pass