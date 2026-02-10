import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import json
import uuid

from backend.core.interfaces.base_protocol import BaseProtocol
from backend.core.models import StreamEvent, EventType, TextData
from backend.core.session.conversation_manager import ConversationManager

# 创建具体子类用于测试 BaseProtocol
class MockProtocolImplementation(BaseProtocol):
    def __init__(self, module_id, config, conversation_manager):
        super().__init__(module_id, config, conversation_manager)
        self.sent_messages = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send_message(self, connection, message):
        self.sent_messages.append((connection, message))

@pytest.fixture
def mock_conversation_manager():
    manager = AsyncMock(spec=ConversationManager)
    # 模拟 conversation_handler 创建
    manager.create_conversation_handler = AsyncMock()
    # 模拟获取 conversation_handler
    manager.get_conversation_handler = MagicMock()
    return manager

@pytest.fixture
def protocol(mock_conversation_manager):
    config = {"host": "test_host", "port": 1234}
    return MockProtocolImplementation("test_protocol", config, mock_conversation_manager)

@pytest.fixture
def mock_connection():
    return MagicMock(name="mock_connection")

@pytest.mark.asyncio
class TestBaseProtocol:

    async def test_handle_register_message(self, protocol, mock_connection, mock_conversation_manager):
        """测试注册消息处理"""
        tag_id = "user123"
        register_event = StreamEvent(
            event_type=EventType.SYSTEM_CLIENT_SESSION_START,
            tag_id=tag_id,
            event_id="evt_1"
        )
        message = register_event.model_dump_json()

        # 模拟 AppContext.get_module
        with patch('src.core.app_context.AppContext.get_module') as mock_get_module:
            await protocol.handle_text_message(mock_connection, message)

        # 验证会话创建
        session_id = protocol.get_session_id(mock_connection)
        assert session_id is not None
        assert protocol.tag_to_session[tag_id] == session_id

        # 验证 ConversationManager 调用
        mock_conversation_manager.create_conversation_handler.assert_called_once()
        call_args = mock_conversation_manager.create_conversation_handler.call_args
        assert call_args.kwargs['session_id'] == session_id
        assert call_args.kwargs['tag_id'] == tag_id
        assert 'send_callback' in call_args.kwargs
        assert 'session_context' in call_args.kwargs

        # 验证响应发送
        assert len(protocol.sent_messages) == 1
        conn, sent_msg = protocol.sent_messages[0]
        assert conn == mock_connection
        response_event = StreamEvent.model_validate_json(sent_msg)
        assert response_event.event_type == EventType.SYSTEM_SERVER_SESSION_START
        assert response_event.session_id == session_id
        assert response_event.tag_id == tag_id

    async def test_handle_text_input_message(self, protocol, mock_connection, mock_conversation_manager):
        """测试文本输入消息处理"""
        # 先建立会话
        tag_id = "user123"
        session_id = protocol.create_session(mock_connection, tag_id)

        # 模拟 handler
        mock_handler = AsyncMock()
        mock_conversation_manager.get_conversation_handler.return_value = mock_handler

        text = "Hello Claude"
        text_event = StreamEvent(
            event_type=EventType.CLIENT_TEXT_INPUT,
            event_id="evt_2",
            event_data=TextData(text=text)
        )
        message = text_event.model_dump_json()

        await protocol.handle_text_message(mock_connection, message)

        # 验证 handler 调用
        mock_conversation_manager.get_conversation_handler.assert_called_with(session_id)
        mock_handler.handle_text_input.assert_called_once_with(text)

    async def test_handle_speech_end_message(self, protocol, mock_connection, mock_conversation_manager):
        """测试语音结束消息处理"""
        # 先建立会话
        session_id = protocol.create_session(mock_connection)

        # 模拟 handler
        mock_handler = AsyncMock()
        mock_conversation_manager.get_conversation_handler.return_value = mock_handler

        # CLIENT_SPEECH_END
        speech_end_event = StreamEvent(
            event_type=EventType.CLIENT_SPEECH_END,
            event_id="evt_3"
        )
        await protocol.handle_text_message(mock_connection, speech_end_event.model_dump_json())
        mock_handler.handle_speech_end.assert_called_once()
        mock_handler.handle_speech_end.reset_mock()

        # STREAM_END
        stream_end_event = StreamEvent(
            event_type=EventType.STREAM_END,
            event_id="evt_4"
        )
        await protocol.handle_text_message(mock_connection, stream_end_event.model_dump_json())
        mock_handler.handle_speech_end.assert_called_once()

    async def test_handle_invalid_message(self, protocol, mock_connection):
        """测试无效消息处理"""
        # 非 JSON 消息
        await protocol.handle_text_message(mock_connection, "not a json")
        # 应该直接返回不抛错（内部捕获或者判断非 { 开头）

        # 格式错误的 JSON
        await protocol.handle_text_message(mock_connection, "{ invalid json }")
        # 内部捕获异常打日志，不抛出

    async def test_handle_audio_message(self, protocol, mock_connection, mock_conversation_manager):
        """测试音频消息处理"""
        audio_data = b'\x01\x02\x03'

        # 未注册连接发送音频 -> 忽略
        await protocol.handle_audio_message(mock_connection, audio_data)
        mock_conversation_manager.get_conversation_handler.assert_not_called()

        # 注册后发送音频
        session_id = protocol.create_session(mock_connection)
        mock_handler = AsyncMock()
        mock_conversation_manager.get_conversation_handler.return_value = mock_handler

        await protocol.handle_audio_message(mock_connection, audio_data)

        mock_conversation_manager.get_conversation_handler.assert_called_with(session_id)
        mock_handler.handle_audio.assert_called_once_with(audio_data)

    async def test_handle_disconnect(self, protocol, mock_connection, mock_conversation_manager):
        """测试断开连接处理"""
        session_id = protocol.create_session(mock_connection)

        await protocol.handle_disconnect(mock_connection)

        # 验证会话移除
        assert protocol.get_session_id(mock_connection) is None

        # 验证 handler 销毁
        mock_conversation_manager.destroy_conversation_handler.assert_called_once_with(session_id)

    async def test_send_event(self, protocol, mock_connection):
        """测试发送事件功能"""
        # 准备会话
        session_id = protocol.create_session(mock_connection)

        event = StreamEvent(
            event_type=EventType.SERVER_TEXT_RESPONSE,
            event_id="evt_resp",
            event_data=TextData(text="response")
        )

        # 成功发送
        success = await protocol.send_event(session_id, event)
        assert success is True
        assert len(protocol.sent_messages) == 1
        assert protocol.sent_messages[0][0] == mock_connection
        # 验证 JSON 结构，而不是字符串，以避免格式差异
        sent_msg = json.loads(protocol.sent_messages[0][1])
        expected_msg = json.loads(event.to_json())
        assert sent_msg == expected_msg

        # 发送给不存在的会话
        success = await protocol.send_event("non_existent_session", event)
        assert success is False

        # 模拟发送失败
        protocol.sent_messages.clear()
        with patch.object(protocol, 'send_message', side_effect=Exception("Send error")):
            success = await protocol.send_event(session_id, event)
            assert success is False

    async def test_session_mapping_logic(self, protocol, mock_connection):
        """测试会话映射的双向一致性"""
        tag_id = "tag_1"
        session_id = protocol.create_session(mock_connection, tag_id)

        # 检查所有映射
        assert protocol.tag_to_session[tag_id] == session_id
        assert protocol.session_to_connection[session_id] == mock_connection
        assert protocol.connection_to_session[mock_connection] == session_id

        # 通过 session_id 移除
        protocol.remove_session(session_id)
        assert len(protocol.tag_to_session) == 0
        assert len(protocol.session_to_connection) == 0
        assert len(protocol.connection_to_session) == 0
