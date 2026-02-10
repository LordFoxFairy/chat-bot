"""ConversationManager 单元测试"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from backend.core.session.conversation_manager import ConversationManager
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.core.session.session_context import SessionContext


@pytest.mark.asyncio
class TestConversationManager:
    """ConversationManager 测试类"""

    def setup_method(self):
        """每个测试前初始化"""
        self.storage = InMemoryStorage()
        self.session_manager = SessionManager(storage_backend=self.storage)
        self.conv_manager = ConversationManager(session_manager=self.session_manager)

    async def test_create_conversation_handler(self):
        """测试创建会话处理器"""
        session_ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag"
        )

        send_callback = AsyncMock()

        # Mock ConversationHandler
        with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockHandler:
            mock_handler = AsyncMock()
            MockHandler.return_value = mock_handler

            handler = await self.conv_manager.create_conversation_handler(
                session_id="test_session",
                tag_id="test_tag",
                send_callback=send_callback,
                session_context=session_ctx
            )

            # 验证创建和启动
            MockHandler.assert_called_once()
            mock_handler.start.assert_called_once()

            # 验证已保存
            assert "test_session" in self.conv_manager.conversation_handlers

    async def test_get_conversation_handler(self):
        """测试获取会话处理器"""
        session_ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag"
        )

        send_callback = AsyncMock()

        with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockHandler:
            mock_handler = AsyncMock()
            MockHandler.return_value = mock_handler

            # 创建
            await self.conv_manager.create_conversation_handler(
                session_id="test_session",
                tag_id="test_tag",
                send_callback=send_callback,
                session_context=session_ctx
            )

            # 获取
            handler = self.conv_manager.get_conversation_handler("test_session")
            assert handler is not None

    async def test_get_nonexistent_handler(self):
        """测试获取不存在的处理器"""
        handler = self.conv_manager.get_conversation_handler("nonexistent")
        assert handler is None

    async def test_destroy_conversation_handler(self):
        """测试销毁会话处理器"""
        session_ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag"
        )

        send_callback = AsyncMock()

        with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockHandler:
            mock_handler = AsyncMock()
            MockHandler.return_value = mock_handler

            # 创建
            await self.conv_manager.create_conversation_handler(
                session_id="test_session",
                tag_id="test_tag",
                send_callback=send_callback,
                session_context=session_ctx
            )

            # 销毁
            await self.conv_manager.destroy_conversation_handler("test_session")

            # 验证 stop 被调用
            mock_handler.stop.assert_called_once()

            # 验证已移除
            assert "test_session" not in self.conv_manager.conversation_handlers

    async def test_destroy_nonexistent_handler(self):
        """测试销毁不存在的处理器"""
        # 不应该抛异常
        await self.conv_manager.destroy_conversation_handler("nonexistent")

    async def test_destroy_all_handlers(self):
        """测试销毁所有处理器"""
        send_callback = AsyncMock()

        with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockHandler:
            mock_handler1 = AsyncMock()
            mock_handler2 = AsyncMock()
            MockHandler.side_effect = [mock_handler1, mock_handler2]

            # 创建多个处理器
            ctx1 = SessionContext(session_id="s1", tag_id="t1")
            ctx2 = SessionContext(session_id="s2", tag_id="t2")

            await self.conv_manager.create_conversation_handler(
                "s1", "t1", send_callback, ctx1
            )
            await self.conv_manager.create_conversation_handler(
                "s2", "t2", send_callback, ctx2
            )

            # 销毁所有
            await self.conv_manager.destroy_all_handlers()

            # 验证所有 stop 被调用
            mock_handler1.stop.assert_called_once()
            mock_handler2.stop.assert_called_once()

            # 验证已清空
            assert len(self.conv_manager.conversation_handlers) == 0

    async def test_create_duplicate_session(self):
        """测试创建重复会话"""
        session_ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag"
        )

        send_callback = AsyncMock()

        with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockHandler:
            mock_handler = AsyncMock()
            MockHandler.return_value = mock_handler

            # 第一次创建
            handler1 = await self.conv_manager.create_conversation_handler(
                "test_session", "test_tag", send_callback, session_ctx
            )

            # 第二次创建相同 session_id，应返回已存在的
            handler2 = await self.conv_manager.create_conversation_handler(
                "test_session", "test_tag", send_callback, session_ctx
            )

            # 应该返回同一个 handler
            assert handler1 is handler2

            # start 只应该被调用一次
            assert mock_handler.start.call_count == 1
