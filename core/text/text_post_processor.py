import re
from typing import List

from core.audio.audio_constants import TextCleaningPatterns


class TextPostProcessor:
    """文本后处理器

    职责:
    - 清洗 ASR 返回的文本
    - 移除特殊标记
    - 合并多个文本段
    """

    def __init__(self):
        self._special_tokens_pattern = re.compile(
            TextCleaningPatterns.SPECIAL_TOKENS_PATTERN
        )

    def clean(self, text: str) -> str:
        """清洗单个文本段

        Args:
            text: 原始文本

        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""

        # 移除特殊标记 (如 FunASR SenseVoice 的 <|tag|>)
        cleaned = self._special_tokens_pattern.sub('', text)

        # 去除首尾空白
        cleaned = cleaned.strip()

        return cleaned

    def merge_segments(self, segments: List[str]) -> str:
        """合并多个文本段

        Args:
            segments: 文本段列表

        Returns:
            str: 合并后的文本
        """
        cleaned_segments = [self.clean(seg) for seg in segments if seg]
        return " ".join(cleaned_segments).strip()
