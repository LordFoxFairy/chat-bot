"""文本数据模型"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class TextData(BaseModel):
    """文本数据模型

    Attributes:
        text: 文本内容
        language: 语言代码 (如 'zh', 'en')
        is_final: 是否为流式文本的最后一部分
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., max_length=100000, description="文本内容")
    message_id: Optional[str] = None
    chunk_id: Optional[str] = None
    language: Optional[str] = Field(None, pattern=r"^[a-z]{2}(-[A-Z]{2})?$")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_final: bool = Field(False, description="是否为最后一部分")

    @model_validator(mode="after")
    def validate_text_content(self) -> "TextData":
        # 如果是最后一部分，允许文本为空（表示结束信号）
        if self.is_final:
            return self

        # 否则文本不能为空且不能全是空格
        if not self.text or not self.text.strip():
            raise ValueError("文本内容不能为空")
        return self

    @property
    def length(self) -> int:
        """文本长度"""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self.text.strip()) == 0

    def truncate(self, max_length: int) -> "TextData":
        """截断文本"""
        if max_length <= 0:
            raise ValueError("最大长度必须大于 0")

        if len(self.text) <= max_length:
            return self

        return self.model_copy(update={"text": self.text[:max_length]})

    @computed_field
    @property
    def display_text(self) -> str:
        """截断后的显示文本"""
        if len(self.text) > 50:
            return self.text[:50] + "..."
        return self.text

    def __str__(self) -> str:
        return f"TextData(text='{self.display_text}', lang={self.language}, final={self.is_final})"

    def __hash__(self) -> int:
        # metadata 是字典，需要转换为可哈希的元组
        metadata_tuple = tuple(sorted(self.metadata.items())) if self.metadata else ()
        return hash((self.text, self.message_id, self.chunk_id, self.language, self.is_final, metadata_tuple))
