import re
from typing import Optional, Pattern

class SentenceSplitter:
    """句子分割器

    职责:
    - 缓存输入的文本块
    - 根据标点符号分割完整句子
    """

    DEFAULT_PATTERN = r'([，。！？；、,.!?;])'

    def __init__(self, delimiter_pattern: Optional[str] = None):
        pattern_str = delimiter_pattern or self.DEFAULT_PATTERN
        self.delimiter_pattern: Pattern = re.compile(pattern_str)
        self.buffer = ""

    def append(self, text: str):
        """追加文本到缓冲区"""
        if text:
            self.buffer += text

    def split(self) -> Optional[str]:
        """尝试分割出一个完整句子

        返回找到的第一个完整句子（包含标点）。
        如果没有完整句子，返回 None。
        """
        match = self.delimiter_pattern.search(self.buffer)
        if match:
            sentence = self.buffer[:match.end()]
            self.buffer = self.buffer[match.end():]
            return sentence
        return None

    def get_remaining(self) -> str:
        """获取并清空剩余缓冲区内容"""
        remaining = self.buffer
        self.buffer = ""
        return remaining

    def clear(self):
        """清空缓冲区"""
        self.buffer = ""
