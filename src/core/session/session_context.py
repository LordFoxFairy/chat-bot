from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

from src.core.app_context import AppContext


class SessionContext(BaseModel):
    """会话上下文

    职责:
    - 存储会话基本信息
    - 支持会话级模块隔离（可覆盖全局模块）
    - 提供统一的模块访问接口
    """

    # 基本信息
    session_id: str = Field(description="会话ID")
    tag_id: str = Field(description="用户唯一标识")

    # 对话历史
    dialogues: List = Field(default_factory=list, description="历史对话")

    # 配置信息
    config: Dict[str, Any] = Field(default_factory=dict, description="会话配置")

    # 会话自定义模块（覆盖全局模块）
    custom_modules: Dict[str, Any] = Field(default_factory=dict, description="会话自定义模块")

    class Config:
        arbitrary_types_allowed = True

    def get_module(self, name: str) -> Optional[Any]:
        """获取模块（先找自定义，再找全局）

        Args:
            name: 模块名称 (vad, asr, llm, tts)

        Returns:
            模块实例，找不到返回 None
        """
        # 优先使用会话自定义模块
        if name in self.custom_modules:
            return self.custom_modules[name]

        # 回退到全局模块
        return AppContext.get_module(name)
