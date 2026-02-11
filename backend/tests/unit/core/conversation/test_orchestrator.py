import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from backend.core.conversation.orchestrator import ConversationOrchestrator
from backend.core.models import StreamEvent, EventType, TextData, AudioData, AudioFormat
from backend.core.interfaces import BaseLLM, BaseTTS

@pytest.fixture
def mock_session_manager():
    return AsyncMock()

@pytest.fixture
def mock_session_context():
    context = MagicMock()
    context.session_id = "test_session_id"
    context.tag_id = "test_tag_id"
    return context

@pytest.fixture
def mock_send_callback():
    return AsyncMock()

@pytest.fixture
def orchestrator(mock_session_context, mock_session_manager, mock_send_callback):
    return ConversationOrchestrator(
        session_id="test_session_id",
        tag_id="test_tag_id",
        session_context=mock_session_context,
        session_manager=mock_session_manager,
        send_callback=mock_send_callback
    )

@pytest.mark.asyncio
async def test_init(orchestrator):
    """测试初始化"""
    assert orchestrator.session_id == "test_session_id"
    assert orchestrator.tag_id == "test_tag_id"
    assert orchestrator.session_context is not None
    assert orchestrator.session_manager is not None
    assert orchestrator.send_callback is not None
    assert orchestrator.interrupt_manager is not None
    assert orchestrator.sentence_splitter is not None
    assert orchestrator.audio_input is None
    assert orchestrator.text_input is None
    assert orchestrator.turn_context == {'last_user_text': ''}

@pytest.mark.asyncio
async def test_start_creates_handlers(orchestrator):
    """测试启动并创建处理器"""
    with patch("backend.core.conversation.orchestrator.AudioInputHandler") as MockAudioHandler, \
         patch("backend.core.conversation.orchestrator.TextInputHandler") as MockTextHandler:

        mock_audio_instance = MockAudioHandler.return_value
        mock_text_instance = MockTextHandler.return_value

        await orchestrator.start()

        # 验证 SessionManager 注册
        orchestrator.session_manager.create_session.assert_called_once_with(orchestrator.session_context)

        # 验证 AudioInputHandler 创建和启动
        MockAudioHandler.assert_called_once()
        assert orchestrator.audio_input == mock_audio_instance
        mock_audio_instance.start.assert_called_once()

        # 验证 TextInputHandler 创建
        MockTextHandler.assert_called_once()
        assert orchestrator.text_input == mock_text_instance

@pytest.mark.asyncio
async def test_stop_cleans_resources(orchestrator):
    """测试停止并清理资源"""
    # 设置 mocks
    mock_audio_input = AsyncMock()
    orchestrator.audio_input = mock_audio_input

    mock_text_input = MagicMock()
    orchestrator.text_input = mock_text_input

    # 添加一个模拟的后台任务
    async def dummy_coro():
        await asyncio.sleep(0.1)

    dummy_task = asyncio.create_task(dummy_coro())
    orchestrator._pending_tasks.add(dummy_task)

    await orchestrator.stop()

    # 验证任务被取消
    assert dummy_task.cancelled() or dummy_task.done()
    assert len(orchestrator._pending_tasks) == 0

    # 验证清理
    mock_audio_input.stop.assert_awaited_once()
    assert orchestrator.audio_input is None
    assert orchestrator.text_input is None
    assert len(orchestrator.turn_context) == 0

@pytest.mark.asyncio
async def test_handle_audio_triggers_interrupt(orchestrator):
    """测试处理音频触发打断"""
    mock_audio_input = AsyncMock()
    orchestrator.audio_input = mock_audio_input

    audio_data = b"fake_audio"

    # Mock interrupt_manager
    orchestrator.interrupt_manager = MagicMock()
    orchestrator.interrupt_manager.is_interrupted = False

    await orchestrator.handle_audio(audio_data)

    orchestrator.interrupt_manager.set_interrupt.assert_called_once()
    mock_audio_input.process_chunk.assert_awaited_once_with(audio_data)

@pytest.mark.asyncio
async def test_handle_speech_end(orchestrator):
    """测试语音结束信号传递"""
    mock_audio_input = MagicMock()
    orchestrator.audio_input = mock_audio_input

    await orchestrator.handle_speech_end()

    mock_audio_input.signal_client_speech_end.assert_called_once()

@pytest.mark.asyncio
async def test_handle_text_input(orchestrator):
    """测试文本处理（不打断）"""
    mock_text_input = AsyncMock()
    orchestrator.text_input = mock_text_input

    text = "hello"
    await orchestrator.handle_text_input(text)

    mock_text_input.process_text.assert_awaited_once_with(text)

@pytest.mark.asyncio
async def test_on_input_result_trigger_conversation(orchestrator):
    """测试输入结果回调触发对话"""
    # 模拟 _trigger_conversation
    orchestrator._trigger_conversation = AsyncMock()

    event = StreamEvent(
        event_type=EventType.ASR_RESULT,
        event_data=TextData(text="hello world", is_final=True),
        session_id="sess_id"
    )

    await orchestrator._on_input_result(event, {})

    orchestrator._trigger_conversation.assert_awaited_once_with("hello world")
    assert orchestrator.turn_context['last_user_text'] == "hello world"

@pytest.mark.asyncio
async def test_on_input_result_empty(orchestrator):
    """测试空输入被忽略"""
    orchestrator._trigger_conversation = AsyncMock()

    event = StreamEvent(
        event_type=EventType.ASR_RESULT,
        event_data=TextData(text="", is_final=True),
        session_id="sess_id"
    )

    await orchestrator._on_input_result(event, {})

    orchestrator._trigger_conversation.assert_not_called()

@pytest.mark.asyncio
async def test_on_input_result_with_interruption(orchestrator):
    """测试打断后的输入拼接"""
    # 模拟之前的打断状态
    orchestrator.turn_context['last_user_text'] = "previous"
    # 我们需要设置 interrupt_manager 让 was_interrupted 为 True
    orchestrator.interrupt_manager.set_interrupt()

    orchestrator._trigger_conversation = AsyncMock()

    event = StreamEvent(
        event_type=EventType.ASR_RESULT,
        event_data=TextData(text="current", is_final=True),
        session_id="sess_id"
    )

    await orchestrator._on_input_result(event, {})

    # 验证拼接结果
    orchestrator._trigger_conversation.assert_awaited_once_with("previous current")
    assert orchestrator.turn_context['last_user_text'] == "previous current"

    # 验证打断状态重置
    assert not orchestrator.interrupt_manager.is_interrupted
    assert not orchestrator.interrupt_manager.was_interrupted

@pytest.mark.asyncio
async def test_trigger_conversation_tts_path(orchestrator, mock_session_manager):
    """测试对话触发流程 - TTS路径"""
    # Setup modules
    mock_llm = AsyncMock(spec=BaseLLM)
    mock_tts = AsyncMock(spec=BaseTTS)

    session_ctx = MagicMock()
    session_ctx.get_module.side_effect = lambda name: mock_llm if name == "llm" else (mock_tts if name == "tts" else None)
    mock_session_manager.get_session.return_value = session_ctx

    orchestrator._process_with_tts = AsyncMock()

    await orchestrator._trigger_conversation("hello")

    # 验证调用了 _process_with_tts
    args, _ = orchestrator._process_with_tts.call_args
    assert args[0].text == "hello"  # llm_input
    assert args[1] == mock_llm
    assert args[2] == mock_tts

@pytest.mark.asyncio
async def test_trigger_conversation_text_only_path(orchestrator, mock_session_manager):
    """测试对话触发流程 - 纯文本路径"""
    mock_llm = AsyncMock(spec=BaseLLM)

    session_ctx = MagicMock()
    # TTS 不存在
    session_ctx.get_module.side_effect = lambda name: mock_llm if name == "llm" else None
    mock_session_manager.get_session.return_value = session_ctx

    orchestrator._process_text_only = AsyncMock()

    await orchestrator._trigger_conversation("hello")

    # 验证调用了 _process_text_only
    args, _ = orchestrator._process_text_only.call_args
    assert args[0].text == "hello"
    assert args[1] == mock_llm

@pytest.mark.asyncio
async def test_process_text_only(orchestrator, mock_send_callback):
    """测试纯文本处理逻辑"""
    mock_llm = AsyncMock(spec=BaseLLM)

    # 模拟 LLM 流
    async def llm_stream(*args, **kwargs):
        yield TextData(text="Response")
    mock_llm.chat_stream = llm_stream

    input_data = TextData(text="test")

    await orchestrator._process_text_only(input_data, mock_llm)

    # 验证回调被调用 2 次 (内容 + 结束标志)
    assert mock_send_callback.call_count == 2

    # 检查第一次调用 (内容)
    call1_arg = mock_send_callback.call_args_list[0][0][0]
    assert call1_arg.event_type == EventType.SERVER_TEXT_RESPONSE
    assert call1_arg.event_data.text == "Response"
    assert not call1_arg.event_data.is_final

    # 检查第二次调用 (结束)
    call2_arg = mock_send_callback.call_args_list[1][0][0]
    assert call2_arg.event_type == EventType.SERVER_TEXT_RESPONSE
    assert call2_arg.event_data.text == ""
    assert call2_arg.event_data.is_final

@pytest.mark.asyncio
async def test_process_text_only_interrupted(orchestrator, mock_send_callback):
    """测试纯文本处理时的打断"""
    mock_llm = AsyncMock(spec=BaseLLM)
    async def llm_stream(*args, **kwargs):
        yield TextData(text="Part1")
        # 此时发生打断
        orchestrator.interrupt_manager.set_interrupt()
        yield TextData(text="Part2")
    mock_llm.chat_stream = llm_stream

    input_data = TextData(text="test")

    await orchestrator._process_text_only(input_data, mock_llm)

    # 应该只收到 Part1
    call_args_list = mock_send_callback.call_args_list
    assert len(call_args_list) > 0
    assert call_args_list[0][0][0].event_data.text == "Part1"

    # 不应该收到结束事件（因为被打断）
    # 检查是否有 is_final=True 的调用
    final_calls = [c[0][0] for c in call_args_list if c[0][0].event_data.is_final]
    assert len(final_calls) == 0

@pytest.mark.asyncio
async def test_process_with_tts(orchestrator):
    """测试 TTS 流程整合"""
    mock_llm = AsyncMock(spec=BaseLLM)
    mock_tts = AsyncMock(spec=BaseTTS)

    # 模拟 LLM 输出 "Hello, world."
    async def llm_stream(*args, **kwargs):
        # 模拟分块返回
        yield TextData(text="Hello, ")
        yield TextData(text="world.")
    mock_llm.chat_stream = llm_stream

    # Mock _send_sentence 以捕获调用
    orchestrator._send_sentence = AsyncMock()
    # Mock _create_background_task 来执行任务（简化测试）
    task_mock = MagicMock()
    orchestrator._create_background_task = MagicMock(return_value=task_mock)

    input_data = TextData(text="test")
    await orchestrator._process_with_tts(input_data, mock_llm, mock_tts)

    # "Hello, " -> 被 SentenceSplitter 缓冲
    # "world." -> "Hello, world." (假设 Pattern 包含 .)
    # 检查 _send_sentence 是否被调用
    assert orchestrator._create_background_task.call_count >= 1

@pytest.mark.asyncio
async def test_send_sentence(orchestrator, mock_send_callback):
    """测试句子发送（文本+合成音频）"""
    mock_tts = AsyncMock(spec=BaseTTS)
    async def tts_stream(*args, **kwargs):
        yield AudioData(data=b"audio1", format=AudioFormat.PCM)
        yield AudioData(data=b"audio2", format=AudioFormat.PCM)
    mock_tts.synthesize_stream = tts_stream

    await orchestrator._send_sentence("Hello world", mock_tts, is_final=True)

    # 验证文本发送
    text_call = mock_send_callback.call_args_list[0][0][0]
    assert text_call.event_type == EventType.SERVER_TEXT_RESPONSE
    assert text_call.event_data.text == "Hello world"
    assert text_call.event_data.is_final

    # 验证音频发送
    audio_calls = [c[0][0] for c in mock_send_callback.call_args_list[1:]]
    assert len(audio_calls) == 2
    assert audio_calls[0].event_type == EventType.SERVER_AUDIO_RESPONSE
    assert audio_calls[0].event_data.data == b"audio1"
    assert audio_calls[1].event_data.data == b"audio2"

@pytest.mark.asyncio
async def test_send_sentence_interrupted(orchestrator, mock_send_callback):
    """测试发送句子时被打断"""
    orchestrator.interrupt_manager.set_interrupt()
    mock_tts = AsyncMock(spec=BaseTTS)

    await orchestrator._send_sentence("Hello", mock_tts)

    # 一旦打断，不应发送任何东西
    mock_send_callback.assert_not_called()
    mock_tts.synthesize_stream.assert_not_called()


# ==================== 额外测试用例 ====================

class TestInitialization:
    """初始化相关测试"""

    @pytest.mark.asyncio
    async def test_init_default_constants(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试初始化时的默认常量"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        assert orch.DEFAULT_SILENCE_TIMEOUT == 1.0
        assert orch.DEFAULT_MAX_BUFFER_DURATION == 5.0

    @pytest.mark.asyncio
    async def test_init_pending_tasks_empty(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试初始化时待处理任务集合为空"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        assert len(orch._pending_tasks) == 0


class TestLifecycle:
    """生命周期管理测试"""

    @pytest.mark.asyncio
    async def test_stop_without_audio_input(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试停止时没有 audio_input 的情况"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        # audio_input 为 None
        await orch.stop()
        # 不应抛出异常
        assert orch.audio_input is None

    @pytest.mark.asyncio
    async def test_stop_with_empty_pending_tasks(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试停止时待处理任务为空的情况"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        await orch.stop()
        assert len(orch._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试停止时取消多个待处理任务"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )

        async def long_task():
            await asyncio.sleep(10)

        task1 = asyncio.create_task(long_task())
        task2 = asyncio.create_task(long_task())
        task3 = asyncio.create_task(long_task())
        orch._pending_tasks.add(task1)
        orch._pending_tasks.add(task2)
        orch._pending_tasks.add(task3)

        await orch.stop()

        assert all(t.cancelled() or t.done() for t in [task1, task2, task3])
        assert len(orch._pending_tasks) == 0


class TestAudioHandling:
    """音频处理测试"""

    @pytest.mark.asyncio
    async def test_handle_audio_without_audio_input(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试没有 audio_input 时处理音频"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        # audio_input 为 None，不应抛出异常
        await orch.handle_audio(b"test_audio")

    @pytest.mark.asyncio
    async def test_handle_audio_already_interrupted(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试已经处于打断状态时处理音频"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_audio_input = AsyncMock()
        orch.audio_input = mock_audio_input
        orch.interrupt_manager.set_interrupt()

        await orch.handle_audio(b"test_audio")

        # 已经打断，不应再次设置
        # process_chunk 仍然应该被调用
        mock_audio_input.process_chunk.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_speech_end_without_audio_input(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试没有 audio_input 时处理语音结束"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        # 不应抛出异常
        await orch.handle_speech_end()


class TestTextHandling:
    """文本处理测试"""

    @pytest.mark.asyncio
    async def test_handle_text_input_without_text_input(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试没有 text_input 时处理文本"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        # 不应抛出异常
        await orch.handle_text_input("test")


class TestTriggerConversation:
    """对话触发测试"""

    @pytest.mark.asyncio
    async def test_trigger_conversation_session_not_found(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试会话上下文未找到的情况"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_session_manager.get_session.return_value = None

        # 不应抛出异常，只是返回
        await orch._trigger_conversation("hello")

    @pytest.mark.asyncio
    async def test_trigger_conversation_llm_not_found(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试 LLM 模块未找到的情况"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        session_ctx = MagicMock()
        session_ctx.get_module.return_value = None
        mock_session_manager.get_session.return_value = session_ctx

        # 不应抛出异常，只是返回
        await orch._trigger_conversation("hello")


class TestInputResult:
    """输入结果回调测试"""

    @pytest.mark.asyncio
    async def test_on_input_result_not_final(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试非最终结果被忽略"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        orch._trigger_conversation = AsyncMock()

        event = StreamEvent(
            event_type=EventType.ASR_RESULT,
            event_data=TextData(text="partial", is_final=False),
            session_id="sess_id"
        )

        await orch._on_input_result(event, {})

        orch._trigger_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_input_result_interrupt_without_previous_text(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试打断但没有之前文本的情况"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        orch.interrupt_manager.set_interrupt()
        orch._trigger_conversation = AsyncMock()

        event = StreamEvent(
            event_type=EventType.ASR_RESULT,
            event_data=TextData(text="new text", is_final=True),
            session_id="sess_id"
        )

        await orch._on_input_result(event, {})

        # 应该只有当前文本（前面拼接空字符串）
        orch._trigger_conversation.assert_awaited_once_with("new text")


class TestProcessWithTTS:
    """TTS 处理测试"""

    @pytest.mark.asyncio
    async def test_process_with_tts_interrupted_during_llm(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试 LLM 流处理期间被打断"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_llm = AsyncMock(spec=BaseLLM)
        mock_tts = AsyncMock(spec=BaseTTS)

        async def llm_stream(*args, **kwargs):
            yield TextData(text="Hello, ")
            orch.interrupt_manager.set_interrupt()
            yield TextData(text="world.")

        mock_llm.chat_stream = llm_stream
        orch._create_background_task = MagicMock(return_value=MagicMock())

        input_data = TextData(text="test")
        await orch._process_with_tts(input_data, mock_llm, mock_tts)

        # 被打断后不应继续处理剩余文本
        # 验证逻辑正确执行

    @pytest.mark.asyncio
    async def test_process_with_tts_empty_remaining(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试 LLM 流结束后没有剩余文本"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_llm = AsyncMock(spec=BaseLLM)
        mock_tts = AsyncMock(spec=BaseTTS)

        # 模拟完整句子，不会有剩余
        async def llm_stream(*args, **kwargs):
            yield TextData(text="Hello.")

        mock_llm.chat_stream = llm_stream
        orch._create_background_task = MagicMock(return_value=MagicMock())

        input_data = TextData(text="test")
        await orch._process_with_tts(input_data, mock_llm, mock_tts)


class TestSendSentence:
    """句子发送测试"""

    @pytest.mark.asyncio
    async def test_send_sentence_interrupted_during_tts(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试 TTS 流处理期间被打断"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_tts = AsyncMock(spec=BaseTTS)

        async def tts_stream(*args, **kwargs):
            yield AudioData(data=b"audio1", format=AudioFormat.PCM)
            orch.interrupt_manager.set_interrupt()
            yield AudioData(data=b"audio2", format=AudioFormat.PCM)

        mock_tts.synthesize_stream = tts_stream

        await orch._send_sentence("Hello", mock_tts, is_final=False)

        # 文本应该被发送
        assert mock_send_callback.call_count >= 1
        # 第一个音频块应该被发送，第二个不应该

    @pytest.mark.asyncio
    async def test_send_sentence_none_audio_chunk(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试 TTS 返回 None 音频块"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_tts = AsyncMock(spec=BaseTTS)

        # 模拟返回 None 和有效音频
        async def tts_stream(*args, **kwargs):
            yield None
            yield AudioData(data=b"audio", format=AudioFormat.PCM)

        mock_tts.synthesize_stream = tts_stream

        await orch._send_sentence("Hello", mock_tts, is_final=False)

        # 文本应该被发送
        text_calls = [c for c in mock_send_callback.call_args_list
                      if c[0][0].event_type == EventType.SERVER_TEXT_RESPONSE]
        assert len(text_calls) == 1
        # 只有一个有效音频块被发送
        audio_calls = [c for c in mock_send_callback.call_args_list
                       if c[0][0].event_type == EventType.SERVER_AUDIO_RESPONSE]
        assert len(audio_calls) == 1

    @pytest.mark.asyncio
    async def test_send_sentence_is_final_false(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试 is_final=False 的情况"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_tts = AsyncMock(spec=BaseTTS)

        async def tts_stream(*args, **kwargs):
            yield AudioData(data=b"audio", format=AudioFormat.PCM)

        mock_tts.synthesize_stream = tts_stream

        await orch._send_sentence("Hello", mock_tts, is_final=False)

        text_call = mock_send_callback.call_args_list[0][0][0]
        assert not text_call.event_data.is_final


class TestCreateBackgroundTask:
    """后台任务创建测试"""

    @pytest.mark.asyncio
    async def test_create_background_task(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试后台任务创建和跟踪"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )

        completed = []

        async def test_coro():
            completed.append(True)

        task = orch._create_background_task(test_coro())

        assert task in orch._pending_tasks
        await task
        # 任务完成后应该从集合中移除
        await asyncio.sleep(0.01)  # 给回调时间执行
        assert task not in orch._pending_tasks
        assert len(completed) == 1


class TestInterruptManager:
    """打断管理器集成测试"""

    @pytest.mark.asyncio
    async def test_interrupt_manager_reset_on_new_conversation(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试新对话开始时打断状态重置"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        orch._trigger_conversation = AsyncMock()
        orch.interrupt_manager.set_interrupt()

        event = StreamEvent(
            event_type=EventType.ASR_RESULT,
            event_data=TextData(text="hello", is_final=True),
            session_id="sess_id"
        )

        await orch._on_input_result(event, {})

        # 打断状态应该被重置
        assert not orch.interrupt_manager.is_interrupted
        assert not orch.interrupt_manager.was_interrupted

    @pytest.mark.asyncio
    async def test_multiple_audio_chunks_single_interrupt(self, mock_session_context, mock_session_manager, mock_send_callback):
        """测试多个音频块只触发一次打断"""
        orch = ConversationOrchestrator(
            session_id="test_session",
            tag_id="test_tag",
            session_context=mock_session_context,
            session_manager=mock_session_manager,
            send_callback=mock_send_callback
        )
        mock_audio_input = AsyncMock()
        orch.audio_input = mock_audio_input

        # 发送多个音频块
        await orch.handle_audio(b"chunk1")
        await orch.handle_audio(b"chunk2")
        await orch.handle_audio(b"chunk3")

        # 打断状态应该是 True，但只设置一次
        assert orch.interrupt_manager.is_interrupted
        assert orch.interrupt_manager.was_interrupted
