"""模型加载和路径解析集成测试

确保模型路径配置正确，模型可以正常加载。
"""

import os
import sys
from pathlib import Path

import pytest

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestModelPaths:
    """测试模型路径配置"""

    @pytest.fixture
    def config(self):
        """加载配置"""
        import asyncio
        from backend.utils.config_loader import ConfigLoader

        config_path = PROJECT_ROOT / "backend" / "configs" / "config.yaml"
        return asyncio.get_event_loop().run_until_complete(
            ConfigLoader.load_config(str(config_path))
        )

    def test_asr_model_path_exists(self, config):
        """测试 ASR 模型路径是否存在"""
        asr_config = config.get("modules", {}).get("asr", {}).get("config", {})
        funasr_config = asr_config.get("funasr_sensevoice", {})
        model_dir = funasr_config.get("model_dir", "")

        # 解析为绝对路径
        if not os.path.isabs(model_dir):
            model_path = PROJECT_ROOT / model_dir
        else:
            model_path = Path(model_dir)

        assert model_path.exists(), f"ASR 模型目录不存在: {model_path}"
        assert model_path.is_dir(), f"ASR 模型路径不是目录: {model_path}"

    def test_vad_model_path_exists(self, config):
        """测试 VAD 模型路径是否存在"""
        vad_config = config.get("modules", {}).get("vad", {}).get("config", {})
        silero_config = vad_config.get("silero_vad", {})
        model_repo_path = silero_config.get("model_repo_path", "")

        # 解析为绝对路径
        if not os.path.isabs(model_repo_path):
            model_path = PROJECT_ROOT / model_repo_path
        else:
            model_path = Path(model_repo_path)

        assert model_path.exists(), f"VAD 模型目录不存在: {model_path}"
        assert model_path.is_dir(), f"VAD 模型路径不是目录: {model_path}"

        # 检查 hubconf.py 存在（Silero VAD 需要）
        hubconf = model_path / "hubconf.py"
        assert hubconf.exists(), f"VAD 模型缺少 hubconf.py: {hubconf}"


class TestModuleInitialization:
    """测试模块初始化"""

    @pytest.fixture
    def config(self):
        """加载配置"""
        import asyncio
        from backend.utils.config_loader import ConfigLoader

        config_path = PROJECT_ROOT / "backend" / "configs" / "config.yaml"
        return asyncio.get_event_loop().run_until_complete(
            ConfigLoader.load_config(str(config_path))
        )

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="需要 FunASR 模型且可能存在 PyTorch 兼容性问题")
    async def test_asr_module_initialization(self, config):
        """测试 ASR 模块初始化"""
        from backend.adapters.asr.asr_factory import asr_registry

        asr_config = config.get("modules", {}).get("asr", {})
        adapter_type = asr_config.get("adapter_type", "funasr_sensevoice")

        # 获取适配器配置
        adapter_config = asr_config.get("config", {}).get(adapter_type, {})

        # 添加通用配置
        adapter_config.update({
            "sample_rate": asr_config.get("config", {}).get(adapter_type, {}).get("sample_rate", 16000),
            "channels": asr_config.get("config", {}).get(adapter_type, {}).get("channels", 1),
            "sample_width": asr_config.get("config", {}).get(adapter_type, {}).get("sample_width", 2),
        })

        # 创建适配器
        adapter = asr_registry.create(adapter_type, "test_asr", adapter_config)
        assert adapter is not None

        # 初始化
        await adapter.setup()
        assert adapter.model is not None

        # 清理
        await adapter.close()

    @pytest.mark.asyncio
    async def test_vad_module_initialization(self, config):
        """测试 VAD 模块初始化"""
        from backend.adapters.vad.vad_factory import vad_registry

        vad_config = config.get("modules", {}).get("vad", {})
        adapter_type = vad_config.get("adapter_type", "silero_vad")

        # 获取适配器配置
        adapter_config = vad_config.get("config", {}).get(adapter_type, {})

        # 创建适配器
        adapter = vad_registry.create(adapter_type, "test_vad", adapter_config)
        assert adapter is not None

        # 初始化
        await adapter.setup()
        # VAD 适配器检查 model 是否加载
        assert adapter.model is not None

        # 清理
        await adapter.close()

    @pytest.mark.asyncio
    async def test_tts_module_initialization(self, config):
        """测试 TTS 模块初始化"""
        from backend.adapters.tts.tts_factory import tts_registry

        tts_config = config.get("modules", {}).get("tts", {})
        adapter_type = tts_config.get("adapter_type", "edge_tts")

        # 获取适配器配置
        adapter_config = tts_config.get("config", {}).get(adapter_type, {})

        # 创建适配器
        adapter = tts_registry.create(adapter_type, "test_tts", adapter_config)
        assert adapter is not None

        # 初始化
        await adapter.setup()
        # TTS 适配器检查 voice 配置
        assert adapter.voice is not None

        # 清理
        await adapter.close()

    @pytest.mark.asyncio
    async def test_llm_module_initialization(self, config):
        """测试 LLM 模块初始化（需要 API Key）"""
        # 检查是否有 API Key
        api_key = os.getenv("API_KEY")
        if not api_key:
            pytest.skip("需要设置 API_KEY 环境变量")

        from backend.adapters.llm.llm_factory import llm_registry

        llm_config = config.get("modules", {}).get("llm", {})
        adapter_type = llm_config.get("adapter_type", "langchain")
        enable_module = llm_config.get("enable_module", "default")

        # 获取适配器配置
        adapter_config = llm_config.get("config", {}).get(enable_module, {})
        adapter_config["system_prompt"] = llm_config.get("system_prompt", "")

        # 创建适配器
        adapter = llm_registry.create(adapter_type, "test_llm", adapter_config)
        assert adapter is not None

        # 初始化
        await adapter.setup()
        # LLM 适配器检查 llm 客户端是否创建
        assert adapter.llm is not None

        # 清理
        await adapter.close()


class TestDownloadScript:
    """测试模型下载脚本"""

    def test_project_root_calculation(self):
        """测试项目根目录计算"""
        script_path = PROJECT_ROOT / "backend" / "scripts" / "download_models.py"
        assert script_path.exists(), f"下载脚本不存在: {script_path}"

        # 验证计算逻辑
        script_dir = script_path.parent
        backend_dir = script_dir.parent
        calculated_root = backend_dir.parent

        assert calculated_root == PROJECT_ROOT

    def test_cache_dir_function(self):
        """测试缓存目录函数"""
        from backend.scripts.download_models import get_cache_dir

        cache_dir = get_cache_dir()
        assert cache_dir.name == "models"
        assert cache_dir.parent.name == "outputs"

    def test_model_directories_exist(self):
        """测试模型目录是否存在"""
        from backend.scripts.download_models import get_cache_dir

        cache_dir = get_cache_dir()

        # ASR 模型目录
        asr_dir = cache_dir / "asr" / "SenseVoiceSmall" / "iic" / "SenseVoiceSmall"
        assert asr_dir.exists(), f"ASR 模型目录不存在: {asr_dir}"

        # VAD 模型目录
        vad_dir = cache_dir / "vad" / "silero-vad"
        assert vad_dir.exists(), f"VAD 模型目录不存在: {vad_dir}"
