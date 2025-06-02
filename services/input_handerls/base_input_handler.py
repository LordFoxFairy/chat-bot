from abc import ABC, abstractmethod
from typing import Dict, Any

from core.event_manager import EventManager
from core.session_manager import SessionManager


class BaseInputHandler(ABC):
    def __init__(self,
                 session_manager: SessionManager,
                 event_manager: EventManager,
                 config: Dict[str, Any]):
        self.session_manager = session_manager
        self.event_manager = event_manager
        self.config = config
        self.is_running = False
        self.active_connections_map: Dict[str, Any] = {}


    @abstractmethod
    async def start(self):
        """启动输入处理器，例如开始监听端口或连接到消息队列。"""
        self.is_running = True
        # print(f"BaseInputHandler ({self.__class__.__name__}): Starting...")
        pass

    @abstractmethod
    async def stop(self):
        """停止输入处理器，释放资源。"""
        self.is_running = False
        # print(f"BaseInputHandler ({self.__class__.__name__}): Stopping...")
        pass

    def register_connection_for_correlation(self, correlation_id: str, connection_details: Any):
        # print(f"BaseInputHandler ({self.__class__.__name__}): Registering connection for CorrID {correlation_id}")
        self.active_connections_map[correlation_id] = connection_details

    def unregister_connection_for_correlation(self, correlation_id: str):
        # print(f"BaseInputHandler ({self.__class__.__name__}): Unregistering connection for CorrID {correlation_id}")
        self.active_connections_map.pop(correlation_id, None)

    def get_connection_for_correlation(self, correlation_id: str) -> Any:
        return self.active_connections_map.get(correlation_id)

