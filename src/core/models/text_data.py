"""文本数据模型"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, computed_field


class TextData(BaseModel):
    """文本数据模型

    Attributes:
        text: 文本内容
        language: 语言代码 (如 'zh', 'en')
        is_final: 是否为流式文本的最后一部分
    """

    text: str = Field(..., max_length=100000, description="文本内容")
    message_id: Optional[str] = None
    chunk_id: Optional[str] = None
    language: Optional[str] = Field(None, pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_final: bool = Field(False, description="是否为最后一部分")

    @computed_field
    @property
    def display_text(self) -> str:
        """截断后的显示文本"""
        if len(self.text) > 50:
            return self.text[:50] + "..."
        return self.text

    def __str__(self) -> str:
        return f"TextData(text='{self.display_text}', lang={self.language}, final={self.is_final})"
