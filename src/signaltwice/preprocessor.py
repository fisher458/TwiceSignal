import polars as pl
from task import missValue, encode, normalizer

#回傳lazy plan
#怎麼同時處裡多個column?，或者自動偵測類型?
#註冊模式

class PreprocessorHandler:
    _preprocessor_strategies = {}

    @classmethod
    def register_preprocessor_strategy(cls, name: str):
        def wrapper(func):
            cls._preprocessor_strategies[name] = func
            return func
        return wrapper
    
    @classmethod
    def get_strategy(cls, name: str):
        """根據名稱取得策略類別"""
        if name not in cls._preprocessor_strategies:
            raise ValueError(f"Strategy '{name}' is not registered.")
        return cls._preprocessor_strategies[name]


@PreprocessorHandler.register_preprocessor_strategy("preprocessNull")
class preprocessorNull:

    def __call__(self, **kwargs) -> pl.LazyFrame:
        pass
@PreprocessorHandler.register_preprocessor_strategy("preprocessEncode")
class preprocessorEncode:

    def __call__(self, **kwargs) -> pl.LazyFrame:
        pass
@PreprocessorHandler.register_preprocessor_strategy("preprocessNormalize")
class preprocessorNormalize:

    def __call__(self, **kwargs) -> pl.LazyFrame:
        pass
@PreprocessorHandler.register_preprocessor_strategy("preprocessMarkup")
class preprocessorMarkup:
    def __call__(self, **kwargs) -> pl.LazyFrame:
        pass
class preprocessor:
    #函數待處理
    def __init__(self, lf: pl.LazyFrame):
        self.lf = lf #接入點
    

    def processNull(self,method) -> pl.LazyFrame:
        #註冊表
        missVal_Handler = missValue.MissValueHandler.get_strategy(method)()
        self.lf = missVal_Handler(self.lf)
        return self 
        

    def processEncode(self,method) -> pl.LazyFrame:
        encode_Handler = encode.encodeHandler.get_strategy(method)()
        self.lf = self.lf.encode_Handler()
   
   
    def processNormalize(self,method) -> pl.LazyFrame:
        normalize_Handler = normalizer.NormalizeHandler.get_strategy(method)()
        self.lf = self.lf.normalize_Handler()

    def processMarkup(self) -> pl.LazyFrame:
        pass



