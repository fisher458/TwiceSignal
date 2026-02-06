from pathlib import Path
import yaml
import json
from typing import Any, Dict



#AI初步寫出，尚未測試！！！
#還需要寫一個參數分配邏輯class


# 1. 輔助類別：負責把 dict 轉成可以用 . 存取的物件
class DictWrapper:
    def __init__(self, data: Dict):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]  # 使用 [] 以便 key 不存在時噴 KeyError
        except KeyError:
            raise AttributeError(f"Config has no attribute '{item}'")
        
        # 核心邏輯：如果是字典，遞迴包裝；否則直接回傳
        if isinstance(value, dict):
            return DictWrapper(value)
        return value
    
    # 支援 print 顯示
    def __repr__(self):
        return repr(self._data)

class Config:
    """單例設定類別"""
    _instance = None
    _initialized = False  # 新增旗標防止重複載入

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 確保只初始化一次 (標準 Python 單例保護)
        if not self._initialized:
            self._data = {}
            self._initialized = True

    def load(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")

        new_data = {}
        if p.suffix in (".yaml", ".yml"):
            with open(p, "r") as f:
                new_data = yaml.safe_load(f) or {} # 防止空檔回傳 None
        elif p.suffix == ".json":
            with open(p, "r") as f:
                new_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {p.suffix}")
        
        # 這裡選擇「更新」而不是「覆蓋」，允許載入多個設定檔 (Base + Override)
        self._data.update(new_data)

    def __getattr__(self, item: str) -> Any:
        # 轉發給 DictWrapper 處理
        return getattr(DictWrapper(self._data), item)

# --- 使用範例 ---
if __name__ == "__main__":
    # 模擬建立一個 yaml 檔
    with open("test_config.yaml", "w") as f:
        yaml.dump({"database": {"host": "localhost", "port": 5432}}, f)

    cfg = Config()
    cfg.load("test_config.yaml")

    # 1. 支援巢狀點號存取
    print(cfg.database.host)  # Output: localhost (原本的程式碼這裡會報錯)
    
    # 2. 測試單例
    cfg2 = Config()
    print(cfg2.database.port) # Output: 5432 (資料還在)

    # 3. 測試錯誤處理
    try:
        print(cfg.database.password)
    except AttributeError as e:
        print(f"Error caught: {e}") # 會清楚告訴你缺了什麼 key