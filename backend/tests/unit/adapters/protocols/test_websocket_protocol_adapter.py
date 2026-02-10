import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import json
import uuid

from backend.adapters.protocols.websocket_protocol_adapter import WebSocketProtocolAdapter
from backend.core.models import StreamEvent, EventType
from backend.core.session.conversation_manager import ConversationManager
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

@pytest.fixture
def mock_conversation_manager():
    manager = AsyncMock(spec=ConversationManager)
    manager.get_conversation_handler.return_value = AsyncMock()
    return manager

@pytest.fixture
def adapter(mock_conversation_manager):
    config = {
        "host": "localhost",
        "port": 8765
    }
    return WebSocketProtocolAdapter("test_ws_protocol", config, mock_conversation_manager)

@pytest.fixture
def mock_websocket():
    ws = AsyncMock(spec=WebSocketServerProtocol)
    # 模拟 async for 循环
    ws.__aiter__.return_value = iter([])
    return ws

@pytest.mark.asyncio
class TestWebSocketProtocolAdapter:

    async def test_initialization(self, adapter):
        """测试初始化和配置"""
        assert adapter.module_id == "test_ws_protocol"
        assert adapter.host == "localhost"
        assert adapter.port == 8765
        assert adapter.server is None

        # 默认配置
        config = {}
        defaults_adapter = WebSocketProtocolAdapter("default", config, AsyncMock())
        assert defaults_adapter.host == "0.0.0.0"
        assert defaults_adapter.port == 8765

    async def test_lifecycle(self, adapter):
        """测试启动和停止"""
        # 模拟 websockets.serve
        mock_server = AsyncMock()
        mock_server.wait_closed = AsyncMock()
        mock_server.close = MagicMock()

        with patch('websockets.serve', new_callable=AsyncMock) as mock_serve:
            mock_serve.return_value = mock_server

            # Start 是一个长时间运行的任务，因为它等待服务器关闭
            # 我们通过让 wait_closed 立即返回来模拟

            start_task = asyncio.create_task(adapter.start())
            await asyncio.sleep(0.1) # 让 start 运行到 wait_closed

            mock_serve.assert_called_once_with(ANY, "localhost", 8765)
            assert adapter.server == mock_server

            # 测试停止
            await adapter.stop()
            adapter.server.close.assert_called_once()
            assert adapter.server.wait_closed.call_count >= 1 # start 和 stop 都会调用

            await start_task

    async def test_session_management(self, adapter, mock_websocket):
        """测试会话管理"""
        tag_id = "user123"

        # 创建会话
        session_id = adapter.create_session(mock_websocket, tag_id)
        assert session_id is not None
        assert isinstance(session_id, str)

        # 验证映射
        assert adapter.get_session_id(mock_websocket) == session_id
        assert adapter.get_connection(session_id) == mock_websocket
        assert adapter.tag_to_session[tag_id] == session_id

        # 移除会话 (通过 session_id)
        adapter.remove_session(session_id)
        assert adapter.get_session_id(mock_websocket) is None
        assert adapter.get_connection(session_id) is None
        assert tag_id not in adapter.tag_to_session

        # 再次创建并测试通过连接移除
        session_id = adapter.create_session(mock_websocket, tag_id)
        adapter.remove_session_by_connection(mock_websocket)
        assert adapter.get_session_id(mock_websocket) is None
        assert adapter.get_connection(session_id) is None
        assert tag_id not in adapter.tag_to_session

        # 测试 clear_all_sessions
        adapter.create_session(mock_websocket, tag_id)
        adapter.clear_all_sessions()
        assert len(adapter.session_to_connection) == 0
        assert len(adapter.connection_to_session) == 0
        assert len(adapter.tag_to_session) == 0

    async def test_send_message(self, adapter, mock_websocket):
        """测试发送消息"""
        message = "test message"

        # 正常发送
        await adapter.send_message(mock_websocket, message)
        mock_websocket.send.assert_called_once_with(message)

        # 连接关闭异常
        mock_websocket.send.reset_mock()
        # 模拟 ConnectionClosed
        # 根据 websockets 版本不同，构造函数可能不同，直接 mock 对象更稳妥

        # websockets 14.0+
        try:
             from websockets.exceptions import ConnectionClosed
             exception = ConnectionClosed(None, None)
        except TypeError:
             # Older versions
             exception = ConnectionClosed(1005, "Closed")

        mock_websocket.send.side_effect = exception

        # 应该捕获异常不抛出
        await adapter.send_message(mock_websocket, message)

        # 其他异常
        mock_websocket.send.reset_mock()
        mock_websocket.send.side_effect = Exception("error")
        # 应该捕获异常不抛出
        await adapter.send_message(mock_websocket, message)

    async def test_handle_client_messages(self, adapter, mock_websocket):
        """测试处理客户端消息流"""
        # 模拟消息流
        messages = [
            '{"event_type": "client.session.start", "event_id": "1", "tag_id": "test_tag"}',
            '{"event_type": "client.text_input", "event_id": "2", "event_data": {"text": "hello"}}'
        ]

        async def message_generator():
            for msg in messages:
                yield msg

        mock_websocket.__aiter__.side_effect = lambda: message_generator()

        # 注意: handle_text_message 和 _handle_register 的单元测试在 test_base_protocol.py 中进行
        # 这里主要测试 _handle_client 是否正确从 websockets 接收消息并传递给通用处理逻辑

        with patch.object(adapter, 'handle_text_message', new_callable=AsyncMock) as mock_handle_text:
            await adapter._handle_client(mock_websocket)

            assert mock_handle_text.call_count == 2
            mock_handle_text.assert_any_call(mock_websocket, messages[0])
            mock_handle_text.assert_any_call(mock_websocket, messages[1])

    async def test_handle_client_disconnect(self, adapter, mock_websocket):
        """测试客户端断开连接的处理"""
        async def empty_generator():
            if False: yield

        mock_websocket.__aiter__.side_effect = lambda: empty_generator()

        with patch.object(adapter, 'handle_disconnect', new_callable=AsyncMock) as mock_handle_disconnect:
            await adapter._handle_client(mock_websocket)
            mock_handle_disconnect.assert_called_once_with(mock_websocket)

    async def test_handle_client_audio(self, adapter, mock_websocket):
        """测试音频数据处理"""
        audio_data = b'\x00\x01\x02'

        async def message_generator():
            yield audio_data

        mock_websocket.__aiter__.side_effect = lambda: message_generator()

        with patch.object(adapter, 'handle_audio_message', new_callable=AsyncMock) as mock_handle_audio:
            await adapter._handle_client(mock_websocket)
            mock_handle_audio.assert_called_once_with(mock_websocket, audio_data)

    async def test_handle_client_error(self, adapter, mock_websocket):
        """测试处理过程中的未捕获异常"""
        mock_websocket.__aiter__.side_effect = Exception("Unexpected error")

        # 确保异常被捕获且连接最后被视为断开
        with patch.object(adapter, 'handle_disconnect', new_callable=AsyncMock) as mock_handle_disconnect:
            await adapter._handle_client(mock_websocket)
            mock_handle_disconnect.assert_called_once_with(mock_websocket)
