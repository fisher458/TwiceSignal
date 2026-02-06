import math
import polars as pl
import pickle
from dataclasses import dataclass, field
from typing import Literal, Any, Dict

# --- 註冊表與工具 ---

class encodeHandler:
    _encodeStrategy = {}

    @classmethod 
    def register_encode_strategy(cls, name: str):
        def wrapper(func):
            cls._encodeStrategy[name] = func
            return func
        return wrapper
    
    @classmethod
    def get_strategy(cls, method: str):
        if method not in cls._encodeStrategy:
            raise ValueError(f"Strategy '{method}' is not registered.")
        return cls._encodeStrategy[method]
    
FrameType = pl.DataFrame | pl.LazyFrame

def _as_lazy(df: FrameType) -> pl.LazyFrame:
    return df.lazy() if isinstance(df, pl.DataFrame) else df

# --- Base Classes ---

class BaseEncoder:
    """Base encoder with fit/transform split and I/O handling."""
    def __init__(self):
        self.column_: str | None = None

    def fit(self, df: FrameType, column: str, **kwargs):
        """Fit the encoder. Must return self."""
        return self

    def transform(self, df: FrameType, column: str, **kwargs) -> FrameType:
        """Transform the data using fitted parameters."""
        raise NotImplementedError

    def fit_transform(self, df: FrameType, column: str, **kwargs) -> FrameType:
        return self.fit(df, column, **kwargs).transform(df, column, **kwargs)

    def __call__(self, df: FrameType, column: str, **kwargs) -> FrameType:
        return self.fit_transform(df, column, **kwargs)

    def _ensure_column(self, column: str | None, encoder_name: str) -> str:
        col = column or self.column_
        if col is None:
            raise ValueError(f"{encoder_name} not fitted. Call fit() first.")
        # 放寬檢查：允許使用者在 transform 時指定不同欄位名稱 (只要資料意義相同)
        # 但若為了嚴謹，保留檢查也是好的
        return col
    
    # [狀態保存實作]
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

class MappingEncoder(BaseEncoder):
    """Base for encoders that map values using a dictionary."""
    def __init__(self, *, suffix: str, unknown_value: Any, dtype: pl.DataType):
        super().__init__()
        self.suffix = suffix
        self.unknown_value = unknown_value
        self.dtype = dtype
        self.mapping_: dict[Any, Any] = {}

    def transform(self, df: FrameType, column: str | None = None, **kwargs) -> FrameType:
        col = self._ensure_column(column, self.__class__.__name__)
        
        # 優化：若 mapping 為空，直接回傳 (避免 Polars 報錯或無效操作)
        if not self.mapping_:
            return df
            
        return df.with_columns(
            pl.col(col)
            .replace(self.mapping_, default=self.unknown_value)
            .cast(self.dtype)
            .alias(f"{col}{self.suffix}")
        )

# --- Pipeline ---

@dataclass
class EncodeStep:
    column: str
    method: str
    init_params: dict[str, Any] = field(default_factory=dict)
    fit_params: dict[str, Any] = field(default_factory=dict)
    transform_params: dict[str, Any] = field(default_factory=dict)

class EncodePipeline:
    def __init__(self, steps: list[EncodeStep]):
        self.steps = steps
        self._encoders: list[tuple[EncodeStep, BaseEncoder]] = []
        self._fitted = False

    def fit(self, df: FrameType):
        # 檢查是否可重用 (簡單緩存機制)
        can_reuse = (
            len(self._encoders) == len(self.steps)
            and all(
                old_step.method == new_step.method
                and old_step.column == new_step.column
                and old_step.init_params == new_step.init_params
                for (old_step, _), new_step in zip(self._encoders, self.steps)
            )
        )
        if not can_reuse:
            self._encoders = []
            for step in self.steps:
                encoder_cls = encodeHandler.get_strategy(step.method)
                # [錯誤訊息優化] 捕捉參數錯誤
                try:
                    encoder = encoder_cls(**step.init_params)
                except TypeError as e:
                    raise TypeError(f"Error initializing '{step.method}': {e}") from e
                self._encoders.append((step, encoder))

        for step, encoder in self._encoders:
            encoder.fit(df, step.column, **step.fit_params)
        self._fitted = True
        return self

    def transform(self, df: FrameType) -> FrameType:
        if not self._fitted:
            raise ValueError("EncodePipeline not fitted. Call fit() first.")
        out = df
        for step, encoder in self._encoders:
            # 注意：某些 encoder (如 OneHot) 預設會 drop 原欄位
            # 若 pipeline 後續步驟依賴該欄位，這裡會報錯。
            # 建議在 EncodeStep 增加 keep_original=True 選項控制
            out = encoder.transform(out, step.column, **step.transform_params)
        return out

# --- Helper ---

class CategoricalLearner:
    def __init__(self, order: Literal["sorted", "appearance"] = "sorted", drop_nulls: bool = True):
        self.order = order
        self.drop_nulls = drop_nulls

    def execute(self, df: FrameType, column: str) -> list:
        lf = _as_lazy(df)
        base = lf.select(pl.col(column))
        if self.drop_nulls:
            base = base.drop_nulls()
        
        # 優化：先 unique 再 sort，減少 sort 負擔
        if self.order == "appearance":
            cats = base.unique(maintain_order=True)
        else:
            cats = base.unique().sort()
            
        return cats.collect().get_column(column).to_list()

# --- Concrete Encoders ---

@encodeHandler.register_encode_strategy("onehot")
class OneHotEncoder(BaseEncoder):
    def __init__(self, order: Literal["sorted", "appearance"] = "sorted"):
        super().__init__()
        self.categories_: list = []
        self.order = order

    def fit(self, df: FrameType, column: str, **kwargs):
        self.column_ = column
        learner = CategoricalLearner(order=self.order, drop_nulls=True)
        self.categories_ = learner.execute(df, column)
        return self

    def transform(self, df: FrameType, column: str | None = None, **kwargs):
        col = self._ensure_column(column, "OneHotEncoder")
        exprs = [
            (pl.col(col) == cat)
            .fill_null(0) # 確保 Null 轉為 0
            .cast(pl.Int8)
            .alias(f"{col}__{cat}")
            for cat in self.categories_
        ]
        return df.drop(col).with_columns(exprs)

@encodeHandler.register_encode_strategy("binary")
class BinaryEncoder(BaseEncoder):
    def __init__(
        self,
        order: Literal["sorted", "appearance"] = "sorted",
        suffix: str = "_bin",
        unknown_value: int = 0, # 注意：0 用於 unknown，所以 mapping 必須從 1 開始
        drop: bool = True,
    ):
        super().__init__()
        self.order = order
        self.suffix = suffix
        self.unknown_value = unknown_value
        self.drop = drop
        self.mapping_: dict[Any, int] = {}
        self.bits_: int = 0

    def fit(self, df: FrameType, column: str, **kwargs):
        learner = CategoricalLearner(order=self.order, drop_nulls=False)
        categories = learner.execute(df, column)
        
        # [關鍵修正]：enumerate 從 1 開始，保留 0 給 unknown_value
        # 這樣 "Unknown" (000) 和 "Category 1" (001) 就不會重疊
        self.mapping_ = {cat: idx + 1 for idx, cat in enumerate(categories)}
        
        # bits 計算要包含最大值 (len + 1)
        max_val = len(categories) + 1
        self.bits_ = max(1, math.ceil(math.log2(max_val)))
        self.column_ = column
        return self

    def transform(self, df: FrameType, column: str | None = None, **kwargs) -> FrameType:
        col = self._ensure_column(column, "BinaryEncoder")
        # 先轉成 Int ID
        base = pl.col(col).replace(self.mapping_, default=self.unknown_value).cast(pl.Int64)
        
        exprs = [
            ((base >> bit) & 1).alias(f"{col}{self.suffix}_{bit}")
            for bit in range(self.bits_)
        ]
        out = df.with_columns(exprs)
        return out.drop(col) if self.drop else out

@encodeHandler.register_encode_strategy("ordinal")
class OrdinalEncoder(MappingEncoder):
    def __init__(
        self,
        order: list[Any] | Literal["sorted", "appearance"] | None = None,
        suffix: str = "_ordinal",
        unknown_value: int | None = None,
    ):
        super().__init__(suffix=suffix, unknown_value=unknown_value, dtype=pl.Int64)
        self.order = order

    def fit(self, df: FrameType, column: str, **kwargs):
        if isinstance(self.order, list):
            categories = list(self.order)
        else:
            order = "sorted" if self.order in (None, "sorted") else "appearance"
            learner = CategoricalLearner(order=order, drop_nulls=False)
            categories = learner.execute(df, column)
        
        self.mapping_ = {cat: idx for idx, cat in enumerate(categories)}
        self.column_ = column
        return self

@encodeHandler.register_encode_strategy("label")
class LabelEncoder(OrdinalEncoder):
    def __init__(
        self,
        order: Literal["sorted", "appearance"] = "sorted",
        suffix: str = "_label",
        unknown_value: int | None = None,
    ):
        super().__init__(order=order, suffix=suffix, unknown_value=unknown_value)

@encodeHandler.register_encode_strategy("frequency")
class FrequencyEncoder(MappingEncoder):
    def __init__(
        self,
        normalize: bool = True,
        suffix: str = "_freq",
        unknown_value: float | None = 0.0, # 頻率未知通常補 0
    ):
        super().__init__(suffix=suffix, unknown_value=unknown_value, dtype=pl.Float64)
        self.normalize = normalize

    def fit(self, df: FrameType, column: str, **kwargs):
        lf = _as_lazy(df)
        # 優化：一次性計算 count 和 total
        stats = lf.group_by(column).len()
        
        if self.normalize:
            total = lf.select(pl.len()).collect().item()
            # 避免 total 為 0
            total = total if total > 0 else 1
            stats = stats.select(pl.col(column), (pl.col("len") / total).alias("val"))
        else:
            stats = stats.select(pl.col(column), pl.col("len").alias("val"))
            
        result = stats.collect()
        self.mapping_ = dict(zip(result.get_column(column), result.get_column("val")))
        self.column_ = column
        return self

@encodeHandler.register_encode_strategy("target")
class TargetEncoder(MappingEncoder):
    def __init__(
        self,
        suffix: str = "_target",
        unknown_value: float | None = None, # 建議 fit 後設為 global mean
        smoothing: float = 10.0,            # [新增] 平滑參數
        min_samples_leaf: int = 1,          # [新增] 最小樣本數
    ):
        super().__init__(suffix=suffix, unknown_value=unknown_value, dtype=pl.Float64)
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.global_mean_: float = 0.0

    def fit(self, df: FrameType, column: str, **kwargs):
        target = kwargs.get("target")
        if target is None:
            raise ValueError("TargetEncoder requires 'target' in fit_params.")
        
        lf = _as_lazy(df)
        
        # 1. 計算 Global Mean
        self.global_mean_ = lf.select(pl.col(target).mean()).collect().item()
        if self.unknown_value is None:
            self.unknown_value = self.global_mean_

        # 2. 計算每個類別的統計量 (Count, Mean)
        stats = (
            lf.group_by(column)
            .agg([
                pl.count(target).alias("count"),
                pl.mean(target).alias("mean")
            ])
            .collect()
        )

        # 3. 應用平滑公式 (Smoothing)
        # formula: (n * mean + smoothing * global) / (n + smoothing)
        # 這裡的 smoothing 參數類似 m-estimate 中的 m
        counts = stats.get_column("count")
        means = stats.get_column("mean")
        
        smoove = (
            (counts * means + self.smoothing * self.global_mean_) / 
            (counts + self.smoothing)
        )
        
        keys = stats.get_column(column).to_list()
        vals = smoove.to_list()
        
        self.mapping_ = dict(zip(keys, vals))
        self.column_ = column
        return self

@encodeHandler.register_encode_strategy("hashing")
class HashingEncoder(BaseEncoder):
    def __init__(
        self,
        num_features: int = 1024,
        seed: int = 0,
        suffix: str = "_hash",
        drop: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.seed = seed
        self.suffix = suffix
        self.drop = drop

    def transform(self, df: FrameType, column: str, **kwargs) -> FrameType:
        col = column
        # Polars 的 hash 是 unstable 的 (跨版本/平台)，若需穩定雜湊需慎選
        hashed = (pl.col(col).hash(seed=self.seed) % self.num_features).alias(f"{col}{self.suffix}")
        out = df.with_columns(hashed)
        return out.drop(col) if self.drop else out

@encodeHandler.register_encode_strategy("count")
class CountEncoder(FrequencyEncoder):
    def __init__(
        self,
        suffix: str = "_count",
        unknown_value: int | None = 0,
    ):
        super().__init__(normalize=False, suffix=suffix, unknown_value=unknown_value)
        self.dtype = pl.Int64