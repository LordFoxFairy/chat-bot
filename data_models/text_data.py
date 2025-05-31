from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import time

class TextData(BaseModel):
    """
    文本数据模型。
    """
    text: str = Field(..., description="实际的文本内容。")
    # source: TextSource = Field(..., description="文本的来源。") # 根据用户要求移除
    language: Optional[str] = Field(None, description="语言代码 (例如 'en', 'zh-CN')。")
    timestamp: Optional[float] = Field(default_factory=time.time, description="文本数据生成的时间戳 (纪元秒)。")
    confidence: Optional[float] = Field(None, description="置信度得分，例如来自 ASR 的结果。")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="任何附加的元数据。")
    chunk_id: Optional[str] = Field(None, description="如果适用，此文本所属流的标识符。")
    is_final: bool = Field(True, description="指示这是否是流式文本的最后一部分 (用于 ASR/LLM 流式处理)。")

    # class Config: # 根据用户要求，如果Config为空则移除
    #     pass

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

        # source_value = self.source.value if self.source else "未知来源" # 根据用户要求移除

        return (f"TextData(文本='{display_text}', " # 移除了来源信息
                f"语言={self.language}, 时间戳={self.timestamp}, 置信度={self.confidence}, "
                f"流ID={self.chunk_id}, 是否最终={self.is_final})")
