"""audio_converter 单元测试"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from backend.core.models import AudioData, AudioFormat


class TestConvertAudioFormat:
    """convert_audio_format 测试类"""

    def test_empty_audio_data_returns_empty_array(self):
        """测试空音频数据返回空数组"""
        import backend.utils.audio_converter as audio_converter

        # 临时修改模块变量
        original_torchaudio = audio_converter.TORCHAUDIO_AVAILABLE
        original_pydub = audio_converter.PYDUB_AVAILABLE

        try:
            audio_converter.TORCHAUDIO_AVAILABLE = False
            audio_converter.PYDUB_AVAILABLE = True

            # 创建模拟 AudioData，绕过 pydantic 验证
            audio = MagicMock()
            audio.data = b""
            audio.format = AudioFormat.WAV
            audio.sample_rate = 16000
            audio.channels = 1
            audio.sample_width = 2

            result = audio_converter.convert_audio_format(
                audio=audio,
                sample_rate=16000,
                channels=1,
                sample_width=2,
                raise_on_error=False
            )
            assert result is not None
            assert len(result) == 0
        finally:
            audio_converter.TORCHAUDIO_AVAILABLE = original_torchaudio
            audio_converter.PYDUB_AVAILABLE = original_pydub

    @patch("src.utils.audio_converter.TORCHAUDIO_AVAILABLE", False)
    @patch("src.utils.audio_converter.PYDUB_AVAILABLE", False)
    def test_no_library_available_returns_none(self):
        """测试无可用库返回 None"""
        # 重新导入模块以应用补丁
        import importlib
        import backend.utils.audio_converter as audio_converter
        importlib.reload(audio_converter)

        audio = AudioData(
            data=b"some_data",
            format=AudioFormat.WAV,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        result = audio_converter.convert_audio_format(
            audio=audio,
            sample_rate=16000,
            channels=1,
            sample_width=2,
            raise_on_error=False
        )
        assert result is None

    @patch("src.utils.audio_converter.TORCHAUDIO_AVAILABLE", False)
    @patch("src.utils.audio_converter.PYDUB_AVAILABLE", False)
    def test_no_library_available_raises_when_flag_set(self):
        """测试无可用库时设置 raise_on_error=True 抛出异常"""
        import importlib
        import backend.utils.audio_converter as audio_converter
        importlib.reload(audio_converter)

        audio = AudioData(
            data=b"some_data",
            format=AudioFormat.WAV,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        with pytest.raises(RuntimeError):
            audio_converter.convert_audio_format(
                audio=audio,
                sample_rate=16000,
                channels=1,
                sample_width=2,
                raise_on_error=True
            )
