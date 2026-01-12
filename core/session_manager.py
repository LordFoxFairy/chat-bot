from typing import Optional, Protocol

from cachetools import LRUCache
from utils.logging_setup import logger


class StorageBackend(Protocol):
    """存储后端协议"""

    def get(self, key: str) -> Optional['SessionContext']: ...

    def set(self, key: str, value: 'SessionContext'): ...

    def close(self): ...


class InMemoryStorage(StorageBackend):
    """内存存储：使用 LRU 缓存"""

    def __init__(self, maxsize: int = 256):
        self._cache = LRUCache(maxsize=maxsize)
        logger.info(f"[SessionStorage] 内存存储初始化，最大容量: {maxsize}")

    def get(self, key: str) -> Optional['SessionContext']:
        return self._cache.get(key)

    def set(self, key: str, value: 'SessionContext'):
        self._cache[key] = value

    def close(self):
        logger.info("[SessionStorage] 关闭内存存储")


class SessionManager:
    """会话管理器

    职责:
    - 管理所有会话的生命周期
    - 提供会话的 CRUD 操作
    - 通过 session_id 作为 key 访问会话
    """

    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend

    def create_session(self, context: 'SessionContext') -> 'SessionContext':
        """创建会话"""
        self.storage.set(context.session_id, context)
        logger.info(f"[SessionManager] 创建会话: {context.session_id}")
        return context

    def get_session(self, session_id: str) -> Optional['SessionContext']:
        """获取会话"""
        return self.storage.get(session_id)

    def close(self):
        """关闭存储"""
        self.storage.close()


# 全局 session_manager 实例
in_memory_backend = InMemoryStorage(maxsize=10000)
session_manager = SessionManager(storage_backend=in_memory_backend)
