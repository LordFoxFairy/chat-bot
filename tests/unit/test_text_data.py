"""TextData 值对象单元测试"""

import pytest
from pydantic import ValidationError

from src.core.models.text_data import TextData


class TestTextDataCreation:
    """测试 TextData 创建"""

    def test_create_with_text_only(self):
        """测试只提供文本创建"""
        text_data = TextData(text="Hello World")
        assert text_data.text == "Hello World"
        assert text_data.language is None

    def test_create_with_language(self):
        """测试带语言创建"""
        text_data = TextData(text="你好", language="zh-CN")
        assert text_data.text == "你好"
        assert text_data.language == "zh-CN"

    def test_create_with_empty_text_raises_error(self):
        """测试空文本抛出异常"""
        with pytest.raises(ValueError, match="文本内容不能为空"):
            TextData(text="")

    def test_create_with_whitespace_text_raises_error(self):
        """测试纯空格文本抛出异常"""
        with pytest.raises(ValueError, match="文本内容不能为空"):
            TextData(text="   ")

    def test_create_with_whitespace_and_content(self):
        """测试包含空格的有效文本可以创建"""
        text_data = TextData(text="  Hello  ")
        assert text_data.text == "  Hello  "

    def test_create_with_empty_text_and_final_is_allowed(self):
        """测试 is_final=True 时允许空文本"""
        text_data = TextData(text="", is_final=True)
        assert text_data.text == ""
        assert text_data.is_final is True


class TestTextDataImmutability:
    """测试 TextData 不可变性"""

    def test_cannot_modify_text(self):
        """测试不能修改文本"""
        text_data = TextData(text="Original")
        with pytest.raises(ValidationError):
            text_data.text = "Modified"

    def test_cannot_modify_language(self):
        """测试不能修改语言"""
        text_data = TextData(text="Hello", language="en")
        with pytest.raises(ValidationError):
            text_data.language = "zh-CN"


class TestTextDataProperties:
    """测试 TextData 属性"""

    def test_length_property(self):
        """测试长度属性"""
        text_data = TextData(text="Hello")
        assert text_data.length == 5

    def test_length_property_with_chinese(self):
        """测试中文长度"""
        text_data = TextData(text="你好世界")
        assert text_data.length == 4

    def test_is_empty_property_false(self):
        """测试非空文本的 is_empty 属性"""
        text_data = TextData(text="Hello")
        # 注意：由于 __post_init__ 验证，不可能创建空文本的 TextData
        # 所以这里测试的是有内容的情况
        assert not text_data.is_empty

    def test_is_empty_with_whitespace_surrounded_text(self):
        """测试包含空格的文本"""
        text_data = TextData(text="  Hello  ")
        assert not text_data.is_empty


class TestTextDataTruncate:
    """测试 TextData 截断方法"""

    def test_truncate_longer_text(self):
        """测试截断长文本"""
        text_data = TextData(text="Hello World")
        truncated = text_data.truncate(5)
        assert truncated.text == "Hello"
        assert truncated.language == text_data.language

    def test_truncate_shorter_text_returns_same_object(self):
        """测试截断长度大于文本时返回相同对象"""
        text_data = TextData(text="Hello")
        truncated = text_data.truncate(10)
        assert truncated is text_data

    def test_truncate_equal_length_returns_same_object(self):
        """测试截断长度等于文本长度时返回相同对象"""
        text_data = TextData(text="Hello")
        truncated = text_data.truncate(5)
        assert truncated is text_data

    def test_truncate_preserves_language(self):
        """测试截断保留语言信息"""
        text_data = TextData(text="你好世界", language="zh-CN")
        truncated = text_data.truncate(2)
        assert truncated.text == "你好"
        assert truncated.language == "zh-CN"

    def test_truncate_with_zero_raises_error(self):
        """测试截断长度为 0 时抛出异常"""
        text_data = TextData(text="Hello")
        with pytest.raises(ValueError, match="最大长度必须大于 0"):
            text_data.truncate(0)

    def test_truncate_with_negative_length_raises_error(self):
        """测试截断长度为负数时抛出异常"""
        text_data = TextData(text="Hello")
        with pytest.raises(ValueError, match="最大长度必须大于 0"):
            text_data.truncate(-1)

    def test_original_object_unchanged_after_truncate(self):
        """测试截断后原对象不变"""
        original = TextData(text="Hello World")
        truncated = original.truncate(5)
        assert original.text == "Hello World"
        assert truncated.text == "Hello"


class TestTextDataEquality:
    """测试 TextData 相等性（值对象特征）"""

    def test_equal_text_data(self):
        """测试相同内容的 TextData 相等"""
        td1 = TextData(text="Hello", language="en")
        td2 = TextData(text="Hello", language="en")
        assert td1 == td2

    def test_different_text_not_equal(self):
        """测试不同文本的 TextData 不相等"""
        td1 = TextData(text="Hello")
        td2 = TextData(text="World")
        assert td1 != td2

    def test_different_language_not_equal(self):
        """测试不同语言的 TextData 不相等"""
        td1 = TextData(text="Hello", language="en")
        td2 = TextData(text="Hello", language="zh-CN")
        assert td1 != td2

    def test_hashable(self):
        """测试 TextData 可哈希（可以作为字典键或集合元素）"""
        td1 = TextData(text="Hello")
        td2 = TextData(text="World")
        text_set = {td1, td2}
        assert len(text_set) == 2
        assert td1 in text_set
        assert td2 in text_set


class TestTextDataRepr:
    """测试 TextData 字符串表示"""

    def test_repr_contains_text(self):
        """测试 repr 包含文本内容"""
        text_data = TextData(text="Hello")
        repr_str = repr(text_data)
        assert "Hello" in repr_str

    def test_repr_contains_language(self):
        """测试 repr 包含语言信息"""
        text_data = TextData(text="Hello", language="en")
        repr_str = repr(text_data)
        assert "en" in repr_str
