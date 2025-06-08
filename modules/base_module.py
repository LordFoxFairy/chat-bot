import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModule(ABC):
    """
    所有模塊的通用基礎類。
    提供模塊ID、配置、事件循環、事件管理器以及初始化和就緒狀態等基礎功能。
    """

    def __init__(self, module_id: str, config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化 BaseModule。

        參數:
            module_id (str): 模塊的唯一標識符。
            config (Optional[Dict[str, Any]]): 模塊的配置字典。
            event_loop (Optional[asyncio.AbstractEventLoop]): 事件循環。
        """
        self.module_id = module_id
        self.config: Dict[str, Any] = config if config is not None else {}
        self.event_loop: asyncio.AbstractEventLoop = event_loop if event_loop is not None else asyncio.get_event_loop()

        self._is_initialized: bool = True
        self._is_ready: bool = True

    @abstractmethod
    async def initialize(self):
        """
        異步初始化模塊資源。
        子類應實現此方法以加載模型、建立連接等。
        成功初始化後，應將 self._is_initialized 和 self._is_ready 設置為 True。
        """
        raise NotImplementedError("子類必須實現 initialize 方法。")

    async def start(self):
        """
        啟動模塊（可選）。
        如果模塊需要在初始化後執行一些啟動操作（例如開始監聽事件），可以覆蓋此方法。
        """
        if not self.is_ready:
            # logger.warning(f"模塊 {self.module_id} 尚未準備就緒，無法啟動。請先調用 initialize。")
            return
        # logger.info(f"模塊 {self.module_id} 已啟動。")
        pass

    async def close(self):
        """
        停止模塊並釋放資源（可選）。
        """
        # logger.info(f"模塊 {self.module_id} 正在停止...")
        self._is_ready = False
        # 添加資源釋放邏輯
        pass

    @property
    def is_initialized(self) -> bool:
        """模塊是否已完成其一次性的初始化設置。"""
        return self._is_initialized

    @property
    def is_ready(self) -> bool:
        """模塊當前是否準備好處理請求。"""
        return self._is_ready

    def update_config(self, new_config: Dict[str, Any]):
        """
        更新模塊的配置。
        子類可以覆蓋此方法以處理特定的配置更新邏輯。
        """
        # logger.info(f"模塊 {self.module_id} 正在更新配置...")
        self.config.update(new_config)
        # 可能需要重新初始化或重置某些狀態
        # self._is_ready = False # 例如，配置更新後可能需要重新檢查就緒狀態
