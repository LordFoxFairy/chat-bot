from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import time


class TextData(BaseModel):
    """
    文本数据模型。
    """
    text: str = Field(..., description="实际的文本内容。")
    message_id: Optional[str] = Field(None, description="消息id")
    chunk_id: Optional[str] = Field(None, description="消息段id")
    language: Optional[str] = Field(None, description="语言代码 (例如 'en', 'zh-CN')。")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="任何附加的元数据。")
    is_final: Optional[bool] = Field(True, description="指示这是否是流式文本的最后一部分 (用于 ASR/LLM 流式处理)。")

    def __str__(self):
        """
        返回 TextData 对象的字符串表示形式。
        """
        # 确保文本在打印时不会过长，同时处理 None 的情况
        display_text = self.text
        if display_text is not None and len(display_text) > 50:
            display_text = display_text[:50] + "..."
        elif display_text is None:
            display_text = "None"

        return (f"TextData(文本='{display_text}', "  # 移除了来源信息
                f"语言={self.language}, 时间戳={self.timestamp} "
                f"是否最终={self.is_final})")
