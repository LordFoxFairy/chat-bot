import pytest
from unittest.mock import Mock, MagicMock
from backend.core.handlers.status_handler import StatusHandler

class TestStatusHandler:
    """测试 StatusHandler 类"""

    def test_init(self):
        """测试 StatusHandler 初始化"""
        mock_provider = Mock()
        handler = StatusHandler(mock_provider)

        assert handler.module_provider == mock_provider

    @pytest.mark.asyncio
    async def test_handle_status_get_all_modules_running(self):
        """测试获取所有模块都在运行时的状态"""
        # 准备
        mock_asr = MagicMock()
        mock_asr.module_id = "asr_1"
        mock_asr.__class__.__name__ = "ASRModule"
        mock_asr._initialized = True

        mock_vad = MagicMock()
        mock_vad.module_id = "vad_1"
        mock_vad.__class__.__name__ = "VADModule"
        mock_vad._initialized = True

        mock_llm = MagicMock()
        mock_llm.module_id = "llm_1"
        mock_llm.__class__.__name__ = "LLMModule"
        mock_llm._initialized = True

        mock_tts = MagicMock()
        mock_tts.module_id = "tts_1"
        mock_tts.__class__.__name__ = "TTSModule"
        mock_tts._initialized = True

        def side_effect(module_type):
            if module_type == 'asr': return mock_asr
            if module_type == 'vad': return mock_vad
            if module_type == 'llm': return mock_llm
            if module_type == 'tts': return mock_tts
            return None

        mock_provider = Mock(side_effect=side_effect)
        handler = StatusHandler(mock_provider)

        # 执行
        result = await handler.handle_status_get()

        # 验证
        assert len(result) == 4

        assert result['asr']['status'] == "running"
        assert result['asr']['module_id'] == "asr_1"
        assert result['asr']['module_type'] == "ASRModule"
        assert result['asr']['initialized'] is True

        assert result['vad']['status'] == "running"
        assert result['llm']['status'] == "running"
        assert result['tts']['status'] == "running"

    @pytest.mark.asyncio
    async def test_handle_status_get_some_modules_stopped(self):
        """测试部分模块停止时的状态"""
        # 准备 - 只有 ASR 和 LLM 运行
        mock_asr = MagicMock()
        mock_asr.module_id = "asr_1"
        mock_asr.__class__.__name__ = "ASRModule"
        mock_asr._initialized = True

        mock_llm = MagicMock()
        mock_llm.module_id = "llm_1"
        mock_llm.__class__.__name__ = "LLMModule"
        mock_llm._initialized = True

        def side_effect(module_type):
            if module_type == 'asr': return mock_asr
            if module_type == 'llm': return mock_llm
            return None # VAD 和 TTS 返回 None

        mock_provider = Mock(side_effect=side_effect)
        handler = StatusHandler(mock_provider)

        # 执行
        result = await handler.handle_status_get()

        # 验证
        assert len(result) == 4

        # 运行的模块
        assert result['asr']['status'] == "running"
        assert result['llm']['status'] == "running"

        # 停止的模块
        assert result['vad']['status'] == "stopped"
        assert result['vad']['error'] == "Not loaded"
        assert 'module_id' not in result['vad']

        assert result['tts']['status'] == "stopped"
        assert result['tts']['error'] == "Not loaded"

    @pytest.mark.asyncio
    async def test_handle_status_get_no_modules(self):
        """测试没有模块运行时的状态"""
        # 准备
        mock_provider = Mock(return_value=None)
        handler = StatusHandler(mock_provider)

        # 执行
        result = await handler.handle_status_get()

        # 验证
        assert len(result) == 4
        expected_modules = ['asr', 'vad', 'llm', 'tts']

        for module in expected_modules:
            assert module in result
            assert result[module]['status'] == "stopped"
            assert result[module]['error'] == "Not loaded"

    @pytest.mark.asyncio
    async def test_module_provider_called_for_each_type(self):
        """测试是否为每种模块类型调用了提供者"""
        # 准备
        mock_provider = Mock(return_value=None)
        handler = StatusHandler(mock_provider)

        # 执行
        await handler.handle_status_get()

        # 验证
        assert mock_provider.call_count == 4

        # 验证调用参数
        from unittest.mock import call
        expected_calls = [
            call('asr'),
            call('vad'),
            call('llm'),
            call('tts')
        ]
        mock_provider.assert_has_calls(expected_calls, any_order=True)

    @pytest.mark.asyncio
    async def test_module_property_fallback(self):
        """测试模块缺少属性时的回退值"""
        # 准备
        mock_module = MagicMock()
        # 删除特定属性以触发 getattr 的默认值
        del mock_module.module_id
        del mock_module._initialized
        mock_module.__class__.__name__ = "TestModule"

        # 模拟 provider 每次都返回这个模块
        mock_provider = Mock(return_value=mock_module)
        handler = StatusHandler(mock_provider)

        # 执行
        result = await handler.handle_status_get()

        # 验证
        # 检查第一个模块（例如 asr）的属性回退
        assert result['asr']['module_id'] == "unknown"
        assert result['asr']['initialized'] is False
        assert result['asr']['module_type'] == "TestModule"
