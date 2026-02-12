import pytest
import asyncio
from unittest.mock import MagicMock, patch

from backend.adapters.tts.edge_tts_adapter import EdgeTTSAdapter
from backend.core.models import TextData, AudioData, AudioFormat
from backend.core.models.exceptions import ModuleInitializationError, ModuleProcessingError

# 模拟 edge_tts 模块
@pytest.fixture
def mock_edge_tts():
    with patch("backend.adapters.tts.edge_tts_adapter.edge_tts") as mock:
        yield mock

# 模拟 EDGE_TTS_AVAILABLE 标志为 True
@pytest.fixture
def mock_edge_tts_available(mock_edge_tts):
    with patch("backend.adapters.tts.edge_tts_adapter.EDGE_TTS_AVAILABLE", True):
        yield mock_edge_tts

@pytest.fixture
def adapter_config():
    return {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+10%",
        "volume": "+20%",
        "pitch": "+5Hz"
    }

@pytest.fixture
def adapter(mock_edge_tts_available, adapter_config):
    return EdgeTTSAdapter("test_edge_tts_module", adapter_config)

@pytest.mark.asyncio
async def test_initialization(adapter):
    """测试初始化和配置解析"""
    assert adapter.voice == "zh-CN-XiaoxiaoNeural"
    assert adapter.rate == "+10%"
    assert adapter.volume == "+20%"
    assert adapter.pitch == "+5Hz"
    assert adapter.output_format == AudioFormat.MP3

@pytest.mark.asyncio
async def test_initialization_defaults(mock_edge_tts_available):
    """测试默认配置初始化"""
    adapter = EdgeTTSAdapter("test_edge_tts_default", {})
    # 验证 BaseTTS 默认值
    assert adapter.voice == "zh-CN-XiaoxiaoNeural"
    # 验证 EdgeTTSAdapter 默认值
    assert adapter.rate == "+0%"
    assert adapter.volume == "+0%"
    assert adapter.pitch == "+0Hz"

def test_initialization_library_not_available():
    """测试 edge-tts 库不可用时的处理"""
    with patch("backend.adapters.tts.edge_tts_adapter.EDGE_TTS_AVAILABLE", False):
        with pytest.raises(ModuleInitializationError, match="edge-tts 库未安装"):
            EdgeTTSAdapter("test_edge_tts_fail", {})

@pytest.mark.asyncio
async def test_setup_success(adapter, mock_edge_tts_available):
    """测试初始化连接检查成功"""
    # 模拟 Communicate 对象及其 stream 方法
    mock_communicate = MagicMock()
    mock_communicate.stream = MagicMock()

    # 模拟 stream 返回一个音频块
    async def valid_stream():
        yield {"type": "audio", "data": b"test_connection_data"}

    mock_communicate.stream.return_value = valid_stream()
    mock_edge_tts_available.Communicate.return_value = mock_communicate

    # 执行初始化
    await adapter.setup()

    # 验证是否创建了用于测试的 Communicate 对象
    mock_edge_tts_available.Communicate.assert_called_with("测试", adapter.voice)
    assert adapter.is_ready

@pytest.mark.asyncio
async def test_setup_timeout_non_fatal(adapter, mock_edge_tts_available):
    """测试初始化连接超时（非致命，只记录警告）"""
    # 模拟 asyncio.timeout 抛出 TimeoutError

    # 定义一个同步函数来返回上下文管理器
    def mock_timeout(delay):
         # 使用一个简单的上下文管理器来模拟超时
         class TimeoutContext:
             async def __aenter__(self):
                 raise asyncio.TimeoutError()
             async def __aexit__(self, exc_type, exc_val, exc_tb):
                 pass
         return TimeoutContext()

    with patch("asyncio.timeout", side_effect=mock_timeout):
        # 新行为：超时不会抛出异常，而是记录警告并标记为就绪
        await adapter.setup()
        # 适配器仍应标记为就绪，以便在首次请求时重试
        assert adapter.is_ready

@pytest.mark.asyncio
async def test_setup_failure_non_fatal(adapter, mock_edge_tts_available):
    """测试初始化连接失败（非致命，只记录警告）"""
    mock_edge_tts_available.Communicate.side_effect = Exception("Connection refused")

    # 新行为：连接失败不会抛出异常，而是记录警告
    await adapter.setup()
    # 适配器仍应标记为就绪，以便在首次请求时重试
    assert adapter.is_ready

@pytest.mark.asyncio
async def test_synthesize_stream(adapter, mock_edge_tts_available):
    """测试语音合成流"""
    # 标记适配器为就绪状态
    adapter._is_ready = True

    # 模拟 Communicate
    mock_communicate = MagicMock()

    # 模拟返回多个音频块和非音频块
    async def mock_stream_generator():
        yield {"type": "audio", "data": b"chunk1"}
        yield {"type": "metadata", "data": "meta"} # 应该被忽略
        yield {"type": "audio", "data": b"chunk2"}

    mock_communicate.stream.return_value = mock_stream_generator()
    mock_edge_tts_available.Communicate.return_value = mock_communicate

    text_data = TextData(text="Hello world")

    # 收集生成的音频数据
    chunks = []
    # 使用 process_text 替代 synthesize_stream 来保证调用链完整性，
    # 但由于在这里我们需要直接测试 synthesize_stream 的输出（含内部 metadata），
    # 我们可以直接调用 synthesize_stream，但要注意 mock 的 adapter.is_ready 状态

    async for chunk in adapter.synthesize_stream(text_data):
        chunks.append(chunk)

    # 验证 Communicate 调用参数
    mock_edge_tts_available.Communicate.assert_called_with(
        "Hello world",
        adapter.voice,
        rate=adapter.rate,
        volume=adapter.volume,
        pitch=adapter.pitch
    )

    # 验证结果
    # 预期: chunk1, chunk2, finish_chunk (empty)
    assert len(chunks) == 3

    # 检查第一个块
    assert chunks[0].data == b"chunk1"
    assert chunks[0].format == AudioFormat.MP3
    assert chunks[0].is_final is False
    assert chunks[0].metadata["chunk_index"] == 0

    # 检查第二个块
    assert chunks[1].data == b"chunk2"
    assert chunks[1].is_final is False
    assert chunks[1].metadata["chunk_index"] == 1

    # 检查结束块
    # 注意：在 AudioData 验证规则中，data 不能为空
    assert chunks[2].data == b" "
    assert chunks[2].is_final is True
    assert chunks[2].metadata["status"] == "complete"
    assert chunks[2].metadata["total_chunks"] == 2

@pytest.mark.asyncio
async def test_synthesize_stream_empty_text(adapter, mock_edge_tts_available):
    """测试空文本处理"""
    adapter._is_ready = True

    # 构造空文本 (is_final=True 允许空文本)
    text_data = TextData(text="", is_final=True)

    chunks = []
    async for chunk in adapter.synthesize_stream(text_data):
        chunks.append(chunk)

    assert len(chunks) == 1
    # 同样的问题，AudioData data 不能为空
    assert chunks[0].data == b" "
    assert chunks[0].is_final is True
    assert chunks[0].metadata["status"] == "empty_input"

    # 验证未调用 Communicate
    mock_edge_tts_available.Communicate.assert_not_called()

@pytest.mark.asyncio
async def test_synthesize_stream_error_after_retries(adapter, mock_edge_tts_available):
    """测试合成过程中错误处理 - 重试耗尽后抛出异常"""
    adapter._is_ready = True
    adapter.max_retries = 3
    adapter.retry_delay = 0.01  # 快速重试以加快测试

    mock_communicate = MagicMock()
    mock_communicate.stream.side_effect = Exception("API Error")
    mock_edge_tts_available.Communicate.return_value = mock_communicate

    text_data = TextData(text="Error case")

    with pytest.raises(ModuleProcessingError, match="合成失败"):
        async for _ in adapter.synthesize_stream(text_data):
            pass

    # 验证重试了 max_retries 次
    assert mock_edge_tts_available.Communicate.call_count == 3


@pytest.mark.asyncio
async def test_synthesize_stream_retry_success(adapter, mock_edge_tts_available):
    """测试重试机制 - 第一次失败，第二次成功"""
    adapter._is_ready = True
    adapter.max_retries = 3
    adapter.retry_delay = 0.01

    call_count = 0

    def mock_communicate_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_comm = MagicMock()

        if call_count == 1:
            # 第一次调用失败
            mock_comm.stream.side_effect = Exception("Temporary error")
        else:
            # 第二次及之后成功
            async def success_stream():
                yield {"type": "audio", "data": b"retry_success"}

            mock_comm.stream.return_value = success_stream()

        return mock_comm

    mock_edge_tts_available.Communicate.side_effect = mock_communicate_factory

    text_data = TextData(text="Retry test")

    chunks = []
    async for chunk in adapter.synthesize_stream(text_data):
        chunks.append(chunk)

    # 验证成功获取到数据
    assert len(chunks) >= 1
    assert any(chunk.data == b"retry_success" for chunk in chunks)
    # 验证调用了 2 次（第一次失败，第二次成功）
    assert call_count == 2

@pytest.mark.asyncio
async def test_metadata_integrity(adapter, mock_edge_tts_available):
    """测试元数据完整性"""
    adapter._is_ready = True

    mock_communicate = MagicMock()
    async def mock_stream():
        yield {"type": "audio", "data": b"data"}

    mock_communicate.stream.return_value = mock_stream()
    mock_edge_tts_available.Communicate.return_value = mock_communicate

    text_data = TextData(text="Test")

    chunks = []
    async for chunk in adapter.synthesize_stream(text_data):
        chunks.append(chunk)

    # 检查最后一块的元数据
    last_chunk = chunks[-1]
    assert last_chunk.is_final is True
    assert "status" in last_chunk.metadata
    assert last_chunk.metadata["status"] == "complete"
    assert "total_chunks" in last_chunk.metadata
    assert isinstance(last_chunk.metadata["total_chunks"], int)
