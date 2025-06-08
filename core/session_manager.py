from typing import Optional, Dict, Any, Protocol

from cachetools import LRUCache

from core.session_context import SessionContext


class StorageBackend(Protocol):
    def get(self, key: str) -> Optional[SessionContext]: ...

    def set(self, key: str, value: SessionContext): ...

    def close(self): ...


# --- 存儲策略實現 (讓它們自己處理序列化) ---

class InMemoryStorage(StorageBackend):
    """內存存儲：直接存儲 SessionContext 物件，無需任何轉換。"""

    def __init__(self, maxsize: int = 256):
        self._cache = LRUCache(maxsize=maxsize)
        print(f"[Storage] InMemory (cachetools) backend is active. Max size: {maxsize}")

    def get(self, key: str) -> Optional[SessionContext]:
        return self._cache.get(key)

    def set(self, key: str, value: SessionContext):
        self._cache[key] = value

    def close(self):
        print("[Storage] InMemory storage has nothing to close.")


# --- 第三部分：極簡化後的 SessionManager ---

class SessionManager:
    """
    一個邏輯純粹的會話管理器，完全不關心存儲和序列化的細節。
    """

    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend

    def create_session(self, context: SessionContext) -> SessionContext:
        """創建會話，直接將物件交給後端處理。"""
        self.storage.set(context.session_id, context)
        print(f"[Manager] 已將會話 {context.session_id} 交給後端存儲。")
        return context

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """從後端獲取會話物件。"""
        print(f"[Manager] 正在從後端查找會話: {session_id}")
        return self.storage.get(session_id)

    def close(self):
        self.storage.close()


in_memory_backend = InMemoryStorage(maxsize=10000)
session_manager = SessionManager(storage_backend=in_memory_backend)
