import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.session.conversation_manager import ConversationManager
from backend.core.session.session_manager import SessionManager
from backend.core.conversation.orchestrator import ConversationOrchestrator

@pytest.fixture
def mock_session_manager():
    return MagicMock(spec=SessionManager)

@pytest.fixture
def manager(mock_session_manager):
    return ConversationManager(session_manager=mock_session_manager)

@pytest.fixture
def mock_send_callback():
    return AsyncMock()

@pytest.fixture
def mock_session_context():
    return MagicMock()

@pytest.mark.asyncio
async def test_init(mock_session_manager):
    """测试初始化"""
    manager = ConversationManager(session_manager=mock_session_manager)
    assert manager.session_manager == mock_session_manager
    assert manager.conversation_handlers == {}
    assert manager._lock is not None

@pytest.mark.asyncio
async def test_create_conversation_handler(manager, mock_send_callback, mock_session_context):
    """测试创建对话处理器"""
    session_id = "test_session"
    tag_id = "test_tag"

    with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockOrchestrator:
        mock_handler = AsyncMock(spec=ConversationOrchestrator)
        MockOrchestrator.return_value = mock_handler

        # Test normal creation
        handler = await manager.create_conversation_handler(
            session_id=session_id,
            tag_id=tag_id,
            send_callback=mock_send_callback,
            session_context=mock_session_context
        )

        assert handler == mock_handler
        assert session_id in manager.conversation_handlers
        assert manager.conversation_handlers[session_id] == handler

        # Verify Orchestrator was initialized correctly
        MockOrchestrator.assert_called_once_with(
            session_id=session_id,
            tag_id=tag_id,
            session_context=mock_session_context,
            session_manager=manager.session_manager,
            send_callback=mock_send_callback
        )

        # Verify start was called
        mock_handler.start.assert_awaited_once()

@pytest.mark.asyncio
async def test_create_duplicate_conversation_handler(manager, mock_send_callback, mock_session_context):
    """测试重复创建对话处理器（幂等性）"""
    session_id = "test_session"
    tag_id = "test_tag"

    with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockOrchestrator:
        mock_handler = AsyncMock(spec=ConversationOrchestrator)
        MockOrchestrator.return_value = mock_handler

        # First creation
        handler1 = await manager.create_conversation_handler(
            session_id=session_id,
            tag_id=tag_id,
            send_callback=mock_send_callback,
            session_context=mock_session_context
        )

        # Second creation with same session_id
        handler2 = await manager.create_conversation_handler(
            session_id=session_id,
            tag_id=tag_id,
            send_callback=mock_send_callback,
            session_context=mock_session_context
        )

        assert handler1 == handler2
        # Orchestrator should only be initialized once
        assert MockOrchestrator.call_count == 1
        # Start should only be called once
        assert mock_handler.start.call_count == 1

@pytest.mark.asyncio
async def test_get_conversation_handler(manager):
    """测试获取对话处理器"""
    session_id = "test_session"
    mock_handler = AsyncMock(spec=ConversationOrchestrator)
    manager.conversation_handlers[session_id] = mock_handler

    # Test getting existing handler
    assert manager.get_conversation_handler(session_id) == mock_handler

    # Test getting non-existent handler
    assert manager.get_conversation_handler("non_existent") is None

@pytest.mark.asyncio
async def test_destroy_conversation_handler(manager):
    """测试销毁单个对话处理器"""
    session_id = "test_session"
    mock_handler = AsyncMock(spec=ConversationOrchestrator)
    manager.conversation_handlers[session_id] = mock_handler

    # Test destroying existing handler
    await manager.destroy_conversation_handler(session_id)

    assert session_id not in manager.conversation_handlers
    mock_handler.stop.assert_awaited_once()

    # Test destroying non-existent handler (should not raise error)
    await manager.destroy_conversation_handler("non_existent")

@pytest.mark.asyncio
async def test_destroy_all_handlers(manager):
    """测试销毁所有对话处理器"""
    handlers = {}
    for i in range(3):
        session_id = f"session_{i}"
        mock_handler = AsyncMock(spec=ConversationOrchestrator)
        handlers[session_id] = mock_handler
        manager.conversation_handlers[session_id] = mock_handler

    assert len(manager.conversation_handlers) == 3

    await manager.destroy_all_handlers()

    assert len(manager.conversation_handlers) == 0
    for handler in handlers.values():
        handler.stop.assert_awaited_once()

@pytest.mark.asyncio
async def test_concurrent_creation(manager, mock_send_callback, mock_session_context):
    """测试并发创建（验证锁机制）"""
    session_id = "concurrent_session"
    tag_id = "test_tag"

    with patch('backend.core.session.conversation_manager.ConversationOrchestrator') as MockOrchestrator:
        mock_handler = AsyncMock(spec=ConversationOrchestrator)
        MockOrchestrator.return_value = mock_handler

        # Simulate race condition by creating multiple tasks
        tasks = [
            manager.create_conversation_handler(
                session_id=session_id,
                tag_id=tag_id,
                send_callback=mock_send_callback,
                session_context=mock_session_context
            )
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All tasks should return the same handler instance
        first_handler = results[0]
        for handler in results[1:]:
            assert handler == first_handler

        # Orchestrator should only be instantiated once
        assert MockOrchestrator.call_count == 1
