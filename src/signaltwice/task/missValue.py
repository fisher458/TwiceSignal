import polars as pl
from abc import ABC, abstractmethod

#註冊表
class MissValueHandler:
    _misValStrategyRegistry = {}

    @classmethod
    def register_misval_strategy(cls, name: str):
        def wrapper(func):
            cls._misValStrategyRegistry[name] = func
            return func
        return wrapper
    
    @classmethod
    def get_strategy(cls, name: str):
        """根據名稱取得策略類別"""
        if name not in cls._misValStrategyRegistry:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return cls._misValStrategyRegistry[name]

class baseMissValueHandler(ABC):
    
    def __init__(self, lf: pl.LazyFrame):
        self.lf = lf

    @abstractmethod
    def __call__(self) -> pl.LazyFrame:
        pass
    

@MissValueHandler.register_misval_strategy("mean")
class MeanMissValueHandler(baseMissValueHandler):
    
    def __call__(self) -> pl.LazyFrame:
        pass

@MissValueHandler.register_misval_strategy("median")
class MedianMissValueHandler(baseMissValueHandler):
    
    def __call__(self) -> pl.LazyFrame:
        pass

@MissValueHandler.register_misval_strategy("mode")
class ModeMissValueHandler(baseMissValueHandler):
    
    def __call__(self) -> pl.LazyFrame:
        pass
   

@MissValueHandler.register_misval_strategy("zero")
class ZeroMissValueHandler(baseMissValueHandler):
    
    def __call__(self) -> pl.LazyFrame:
        pass

@MissValueHandler.register_misval_strategy("drop")
class DropMissValueHandler(baseMissValueHandler):
    
    def __call__(self) -> pl.LazyFrame:
        pass
