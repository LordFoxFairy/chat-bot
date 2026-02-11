import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import numpy as np

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from backend.core.models.audio_data import AudioData, AudioFormat
from backend.core.models.exceptions import ModuleInitializationError, ModuleProcessingError
from backend.adapters.asr.funasr_sensevoice_adapter import FunASRSenseVoiceAdapter, FUNASR_AVAILABLE

# 模拟 AudioData
@pytest.fixture
def mock_audio_data():
    return AudioData(
        data=b'\x00\x00' * 16000,
        sample_rate=16000,
        sample_width=2,
        channels=1,
        format=AudioFormat.PCM
    )

# 模拟配置
@pytest.fixture
def mock_config():
    return {
        "model_dir": "/tmp/mock_model_dir",
        "device": "cpu",
        "vad_chunk_size": 5000,
        "output_dir": "/tmp/mock_output"
    }

# 如果没有安装funasr库，我们需要mock这个导入
# 注意：在 adapter 模块加载时，如果 funasr 不存在，会有全局变量 FUNASR_AVAILABLE = False
# 为了测试 FunASR 相关的逻辑，我们需要 patch sys.modules 或在测试中 mock 那个模块
@pytest.fixture
def mock_funasr_module():
    with patch.dict(sys.modules, {'funasr': MagicMock()}):
        yield

class TestFunASRSenseVoiceAdapter:

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    async def test_initialization_success(self, mock_auto_model, mock_resolve_path, mock_config):
        """测试正常初始化"""
        # 设置
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_model_instance = MagicMock()
        mock_auto_model.return_value = mock_model_instance

        # 执行
        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)
        await adapter.setup()

        # 验证
        assert adapter.model is not None
        assert adapter.model == mock_model_instance

        # 验证 AutoModel 调用参数
        mock_auto_model.assert_called_once()
        call_kwargs = mock_auto_model.call_args.kwargs
        assert call_kwargs['device'] == mock_config['device']
        assert call_kwargs['chunk_size'][0] == mock_config['vad_chunk_size']
        assert 'output_dir' in call_kwargs

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', False)
    async def test_init_library_not_installed(self, mock_config):
        """测试库未安装时的初始化"""
        with pytest.raises(ModuleInitializationError, match="funasr 库未安装"):
            FunASRSenseVoiceAdapter("test_asr", mock_config)

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    async def test_init_model_dir_not_found(self, mock_resolve_path, mock_config):
        """测试模型目录不存在"""
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_resolve_path.return_value = mock_path

        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)

        with pytest.raises(ModuleInitializationError, match="模型目录不存在"):
            await adapter.setup()

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    async def test_init_automodel_failure(self, mock_auto_model, mock_resolve_path, mock_config):
        """测试 AutoModel 初始化失败"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_auto_model.side_effect = Exception("Model load failed")

        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)

        with pytest.raises(ModuleInitializationError, match="FunASR 初始化失败"):
            await adapter.setup()

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.convert_audio_format')
    async def test_recognize_success(self, mock_convert, mock_auto_model, mock_resolve_path, mock_config, mock_audio_data):
        """测试正常的语音识别流程"""
        # 设置
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_model_instance = MagicMock()
        # 模拟 generate 方法
        mock_model_instance.generate.return_value = [{"text": "测试语音识别"}]
        mock_auto_model.return_value = mock_model_instance

        # 模拟音频转换
        mock_audio_array = np.zeros(100, dtype=np.float32)
        mock_convert.return_value = mock_audio_array

        # 初始化
        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)
        await adapter.setup()

        # 执行
        result = await adapter.recognize(mock_audio_data)

        # 验证
        assert result == "测试语音识别"
        mock_convert.assert_called_once()
        # 验证 generate 调用
        mock_model_instance.generate.assert_called_once()
        call_kwargs = mock_model_instance.generate.call_args.kwargs
        # numpy 数组比较需要特殊处理，这里简单验证参数存在
        assert 'input' in call_kwargs
        assert 'fs' in call_kwargs

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    async def test_recognize_uninitialized(self, mock_auto_model, mock_resolve_path, mock_config, mock_audio_data):
        """测试未初始化直接调用识别"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_resolve_path.return_value = mock_path

        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)
        # 不调用 initialize

        with pytest.raises(ModuleProcessingError, match="模型未初始化"):
            await adapter.recognize(mock_audio_data)

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.convert_audio_format')
    async def test_recognize_preprocess_fail(self, mock_convert, mock_auto_model, mock_resolve_path, mock_config, mock_audio_data):
        """测试预处理失败（返回空）"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_auto_model.return_value = MagicMock()
        mock_convert.return_value = None  # 转换失败返回 None 或空数组

        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)
        await adapter.setup()

        result = await adapter.recognize(mock_audio_data)
        assert result == ""

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.convert_audio_format')
    async def test_recognize_inference_error(self, mock_convert, mock_auto_model, mock_resolve_path, mock_config, mock_audio_data):
        """测试推理过程出错"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_model = MagicMock()
        mock_model.generate.side_effect = Exception("Inference error")
        mock_auto_model.return_value = mock_model

        mock_convert.return_value = np.zeros(100, dtype=np.float32)

        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)
        await adapter.setup()

        with pytest.raises(ModuleProcessingError, match="推理失败"):
            await adapter.recognize(mock_audio_data)

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    async def test_text_extraction(self, mock_auto_model, mock_resolve_path, mock_config):
        """测试文本提取逻辑"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_auto_model.return_value = MagicMock()

        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)
        await adapter.setup()

        # 测试正常结果 list[dict]
        res1 = adapter._extract_text([{"text": "你好"}, {"text": "世界"}])
        assert res1 == "你好 世界"

        # 测试空结果
        res2 = adapter._extract_text([])
        assert res2 == ""

        # 测试 None
        res3 = adapter._extract_text(None)
        assert res3 == ""

        # 测试非列表结果
        res4 = adapter._extract_text({})
        assert res4 == ""

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    async def test_cleanup(self, mock_auto_model, mock_resolve_path, mock_config):
        """测试资源清理"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        adapter = FunASRSenseVoiceAdapter("test_asr", mock_config)
        await adapter.setup()

        assert adapter.model is not None

        await adapter.close()

        assert adapter.model is None

    @pytest.mark.asyncio
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.FUNASR_AVAILABLE', True)
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.resolve_project_path')
    @patch('backend.adapters.asr.funasr_sensevoice_adapter.AutoModel')
    async def test_cleanup_cuda(self, mock_auto_model, mock_resolve_path, mock_config):
        """测试 CUDA 资源清理"""
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/resolved/path")
        mock_resolve_path.return_value = mock_path
        mock_auto_model.return_value = MagicMock()

        config_cuda = mock_config.copy()
        config_cuda["device"] = "cuda"

        adapter = FunASRSenseVoiceAdapter("test_asr", config_cuda)
        await adapter.setup()

        # 模拟 torch
        with patch.dict(sys.modules, {'torch': MagicMock()}):
            mock_torch = sys.modules['torch']
            mock_torch.cuda.is_available.return_value = True

            await adapter.close()

            mock_torch.cuda.empty_cache.assert_called_once()
