from typing import Any, Dict, Optional, List,TYPE_CHECKING

from pydantic import BaseModel, Field


class SessionContext(BaseModel):
    """
    優化後的會話上下文模型。

    用於存儲和傳遞與單個用戶會話相關的狀態、配置和依賴模塊。
    """
    tag_id: Optional[str] = Field(default=None, description="用戶唯一標識。")
    session_id: Optional[str] = Field(default=None, description="用戶會話ID，通常在會話開始時生成。")
    dialogues: Optional[List] = Field(default=None, description="歷史對話")

    config: Dict[str, Any] = Field(default_factory=dict,
                                   description="配置信息，例如 API keys 或功能開關。"
                                               "使用 default_factory=dict 避免所有實例共享同一個字典。")

    global_module_manager: Optional[Any] = Field(default=None, description="全局模塊管理器，在整個應用生命週期內共享。")

    module_manager: Optional[Any] = Field(default=None,description="當前會話特定的模塊管理器。")

    class Config:
        # 允許欄位是自訂類物件
        arbitrary_types_allowed = True


SessionContext.model_rebuild()
