"""SileroVADAdapter 单元测试

使用 mock 来避免对 torch 的依赖
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys

# 创建完整的 torch mock
mock_torch = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.cuda.empty_cache = MagicMock()
mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

# Mock tensor
mock_tensor = MagicMock()
mock_tensor.dim.return_value = 1
mock_tensor.shape = [512]
mock_torch.from_numpy = MagicMock(return_value=mock_tensor)
mock_tensor.to = MagicMock(return_value=mock_tensor)

sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch.cuda

import numpy as np

# 现在可以安全导入
from backend.adapters.vad.silero_vad_adapter import SileroVADAdapter
from backend.core.models.exceptions import ModuleInitializationError, ModuleProcessingError


@pytest.fixture
def vad_config():
    return {
        "model_repo_path": "snakers4/silero-vad",
        "model_name": "silero_vad",
        "threshold": 0.5,
        "sample_rate": 16000,
        "device": "cpu",
        "window_size_samples": 512,
        "force_reload_model": False,
        "max_consecutive_failures": 3
    }


@pytest.fixture
def vad_adapter(vad_config):
    return SileroVADAdapter("test_vad", vad_config)


class TestSileroVADAdapter:

    def test_initialization(self, vad_config):
        """测试初始化和配置解析"""
        adapter = SileroVADAdapter("test_vad", vad_config)

        assert adapter.module_id == "test_vad"
        assert adapter.model_repo_path == "snakers4/silero-vad"
        assert adapter.model_name == "silero_vad"
        assert adapter.threshold == 0.5
        assert adapter.sample_rate == 16000
        assert adapter.device == "cpu"
        assert adapter.window_size_samples == 512
        assert adapter.max_consecutive_failures == 3
        assert adapter.model is None

    @pytest.mark.asyncio
    async def test_setup_success(self, vad_adapter):
        """测试模型成功加载"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        with patch.object(mock_torch.hub, 'load', return_value=mock_model):
            await vad_adapter.setup()

            assert vad_adapter.is_ready
            assert vad_adapter.model == mock_model
            mock_model.to.assert_called_with("cpu")
            mock_model.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_failure(self, vad_adapter):
        """测试模型加载失败"""
        with patch.object(mock_torch.hub, 'load', side_effect=Exception("Download failed")):
            with pytest.raises(ModuleInitializationError) as excinfo:
                await vad_adapter.setup()

            assert "Silero VAD 初始化失败" in str(excinfo.value)
            assert not vad_adapter.is_ready

    @pytest.mark.asyncio
    async def test_setup_model_tuple_return(self, vad_adapter):
        """测试 torch.hub.load 返回元组的情况"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        with patch.object(mock_torch.hub, 'load', return_value=(mock_model, "utils")):
            await vad_adapter.setup()
            assert vad_adapter.model == mock_model

    @pytest.mark.asyncio
    async def test_setup_model_none_return(self, vad_adapter):
        """测试 torch.hub.load 返回 None"""
        with patch.object(mock_torch.hub, 'load', return_value=None):
            with pytest.raises(ModuleInitializationError) as excinfo:
                await vad_adapter.setup()

            assert "torch.hub.load 返回 None" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_detect_not_initialized(self, vad_adapter):
        """测试未初始化时调用 detect"""
        with pytest.raises(ModuleProcessingError) as excinfo:
            await vad_adapter.detect(b"audio_bytes")

        assert "模型未初始化" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_detect_empty_data(self, vad_adapter):
        """测试空音频数据"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        with patch.object(mock_torch.hub, 'load', return_value=mock_model):
            await vad_adapter.setup()

        result = await vad_adapter.detect(b"")
        assert result is False

    @pytest.mark.asyncio
    async def test_detect_speech_detected(self, vad_adapter):
        """测试检测到语音"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        # 模型推理返回高概率
        mock_prob = MagicMock()
        mock_prob.item.return_value = 0.8
        mock_model.return_value = mock_prob

        # 设置正确的 tensor mock - 必须设置 ndim 属性
        local_tensor = MagicMock()
        local_tensor.ndim = 1  # 直接设置属性而不是 return_value
        local_tensor.shape = [512]
        local_tensor.to.return_value = local_tensor

        with patch.object(mock_torch.hub, 'load', return_value=mock_model), \
             patch.object(mock_torch, 'from_numpy', return_value=local_tensor), \
             patch.object(mock_torch, 'no_grad', return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())):

            await vad_adapter.setup()

            audio_data = np.zeros(512, dtype=np.int16).tobytes()
            result = await vad_adapter.detect(audio_data)

            assert result is True
            assert vad_adapter.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_detect_no_speech(self, vad_adapter):
        """测试未检测到语音"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        # 模型推理返回低概率
        mock_prob = MagicMock()
        mock_prob.item.return_value = 0.2
        mock_model.return_value = mock_prob

        local_tensor = MagicMock()
        local_tensor.ndim = 1  # 直接设置属性
        local_tensor.shape = [512]
        local_tensor.to.return_value = local_tensor

        with patch.object(mock_torch.hub, 'load', return_value=mock_model), \
             patch.object(mock_torch, 'from_numpy', return_value=local_tensor), \
             patch.object(mock_torch, 'no_grad', return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())):

            await vad_adapter.setup()

            audio_data = np.zeros(512, dtype=np.int16).tobytes()
            result = await vad_adapter.detect(audio_data)

            assert result is False

    @pytest.mark.asyncio
    async def test_consecutive_failures(self, vad_adapter):
        """测试连续失败计数"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        local_tensor = MagicMock()
        local_tensor.ndim = 1  # 直接设置属性
        local_tensor.shape = [512]
        local_tensor.to.return_value = local_tensor

        # 模型推理时抛出异常
        mock_model.side_effect = Exception("Inference error")

        with patch.object(mock_torch.hub, 'load', return_value=mock_model), \
             patch.object(mock_torch, 'from_numpy', return_value=local_tensor), \
             patch.object(mock_torch, 'no_grad', return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())):

            await vad_adapter.setup()
            audio_data = np.zeros(512, dtype=np.int16).tobytes()

            # 失败后返回 False
            result = await vad_adapter.detect(audio_data)
            assert result is False
            assert vad_adapter.consecutive_failures == 1

            result = await vad_adapter.detect(audio_data)
            assert result is False
            assert vad_adapter.consecutive_failures == 2

            # 第3次失败抛出异常
            with pytest.raises(ModuleProcessingError) as excinfo:
                await vad_adapter.detect(audio_data)
            assert "连续失败 3 次" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_reset_state(self, vad_adapter):
        """测试状态重置"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.reset_states = MagicMock()

        with patch.object(mock_torch.hub, 'load', return_value=mock_model):
            await vad_adapter.setup()
            await vad_adapter.reset_state()

            mock_model.reset_states.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, vad_adapter):
        """测试资源清理"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        with patch.object(mock_torch.hub, 'load', return_value=mock_model):
            await vad_adapter.setup()
            assert vad_adapter.model is not None

            await vad_adapter.close()

            assert vad_adapter.model is None
            assert not vad_adapter.is_ready

    @pytest.mark.asyncio
    async def test_close_with_cuda(self, vad_adapter):
        """测试 CUDA 设备时的资源清理"""
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        with patch.object(mock_torch.hub, 'load', return_value=mock_model):
            await vad_adapter.setup()
            vad_adapter.device = "cuda"

            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.empty_cache.reset_mock()

            await vad_adapter.close()

            mock_torch.cuda.empty_cache.assert_called_once()
