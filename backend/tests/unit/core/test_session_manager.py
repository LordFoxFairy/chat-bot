"""SessionManager 单元测试"""
import pytest
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.core.session.session_context import SessionContext


class TestInMemoryStorage:
    """InMemoryStorage 测试类"""

    def test_set_and_get(self):
        """测试存储和获取"""
        storage = InMemoryStorage(maxsize=10)
        ctx = SessionContext(session_id="test_1", tag_id="user_1")

        storage.set("test_1", ctx)
        result = storage.get("test_1")

        assert result is not None
        assert result.session_id == "test_1"

    def test_get_nonexistent(self):
        """测试获取不存在的会话"""
        storage = InMemoryStorage()
        result = storage.get("nonexistent")
        assert result is None

    def test_lru_eviction(self):
        """测试 LRU 淘汰"""
        storage = InMemoryStorage(maxsize=2)

        ctx1 = SessionContext(session_id="s1", tag_id="u1")
        ctx2 = SessionContext(session_id="s2", tag_id="u2")
        ctx3 = SessionContext(session_id="s3", tag_id="u3")

        storage.set("s1", ctx1)
        storage.set("s2", ctx2)
        storage.set("s3", ctx3)  # 应该淘汰 s1

        # s1 应该被淘汰
        assert storage.get("s1") is None
        assert storage.get("s2") is not None
        assert storage.get("s3") is not None


@pytest.mark.asyncio
class TestSessionManager:
    """SessionManager 测试类"""

    async def test_create_session(self):
        """测试创建会话"""
        storage = InMemoryStorage()
        manager = SessionManager(storage_backend=storage)

        ctx = SessionContext(session_id="test_session", tag_id="test_user")
        result = await manager.create_session(ctx)

        assert result.session_id == "test_session"
        assert result.tag_id == "test_user"

    async def test_get_session(self):
        """测试获取会话"""
        storage = InMemoryStorage()
        manager = SessionManager(storage_backend=storage)

        ctx = SessionContext(session_id="test_session", tag_id="test_user")
        await manager.create_session(ctx)

        # 获取会话
        result = await manager.get_session("test_session")
        assert result is not None
        assert result.session_id == "test_session"

    async def test_get_nonexistent_session(self):
        """测试获取不存在的会话"""
        storage = InMemoryStorage()
        manager = SessionManager(storage_backend=storage)

        result = await manager.get_session("nonexistent")
        assert result is None

    def test_close(self):
        """测试关闭存储"""
        storage = InMemoryStorage()
        manager = SessionManager(storage_backend=storage)

        # 不应该抛异常
        manager.close()
