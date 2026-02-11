from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock
from backend.core.app_context import AppContext
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.utils.config_loader import ConfigLoader
from backend.core.interfaces.base_llm import BaseLLM
from backend.core.interfaces.base_tts import BaseTTS

@pytest.fixture(autouse=True)
def clean_app_context():
    """每个测试前后自动清理 AppContext"""
    # Setup - ensure clean state before test
    AppContext.clear()

    yield

    # Teardown - ensure clean state after test
    AppContext.clear()

@pytest.fixture
def session_manager():
    """创建 InMemoryStorage 和 SessionManager"""
    storage = InMemoryStorage()
    manager = SessionManager(storage_backend=storage)
    yield manager
    # Teardown
    manager.close()

@pytest.fixture
async def test_config():
    """加载 backend/configs/test_config.yaml 配置（如果存在）"""
    config_path = Path(__file__).parent / ".." / "configs" / "test_config.yaml"
    if config_path.exists():
        return await ConfigLoader.load_config(str(config_path))

    # Return a default minimal valid config structure if file doesn't exist
    # This ensures tests depending on this fixture don't crash without the file
    return {
        "modules": {
            "asr": {
                "enabled": True,
                "module_category": "asr",
                "adapter_type": "mock",
                "enable_module": "mock",
                "config": {}
            },
            "llm": {
                "enabled": True,
                "module_category": "llm",
                "adapter_type": "mock",
                "enable_module": "mock",
                "config": {
                    "mock": {
                         "model_name": "mock-model"
                    }
                }
            },
            "tts": {
                "enabled": True,
                "module_category": "tts",
                "adapter_type": "mock",
                "config": {
                    "mock": {}
                }
            },
            "vad": {
                "enabled": True,
                "module_category": "vad",
                "adapter_type": "mock",
                "config": {}
            }
        },
        "logging": {
            "level": "DEBUG"
        }
    }

@pytest.fixture
def mock_llm_module():
    """创建 LLM 模块的 AsyncMock"""
    mock_llm = AsyncMock(spec=BaseLLM)
    mock_llm.module_id = "test_llm"
    mock_llm.config = {"model_name": "test-model"}
    mock_llm.is_ready = True

    # Mock chat_stream as an async generator
    async def mock_stream(*args, **kwargs):
        # Yield a mock TextData
        from backend.core.models import TextData
        yield TextData(text="Mock LLM Response", is_final=True, chunk_id="test")

    mock_llm.chat_stream = mock_stream
    mock_llm.process_text = mock_stream

    return mock_llm

@pytest.fixture
def mock_tts_module():
    """创建 TTS 模块的 AsyncMock"""
    mock_tts = AsyncMock(spec=BaseTTS)
    mock_tts.module_id = "test_tts"
    mock_tts.config = {"voice": "test-voice"}
    mock_tts.is_ready = True

    # Mock synthesize_stream as an async generator
    async def mock_stream(*args, **kwargs):
        # Yield a mock AudioData
        from backend.core.models import AudioData, AudioFormat
        yield AudioData(data=b"mock audio", format=AudioFormat.WAV, is_final=True)

    mock_tts.synthesize_stream = mock_stream
    mock_tts.process_text = mock_stream

    return mock_tts
