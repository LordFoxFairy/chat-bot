from abc import ABC, abstractmethod
from typing import Dict, Any
from utils.logging_setup import logger

class BaseHandler(ABC):
    """
    基础输入处理器的抽象基类。
    定义了处理不同类型输入（音频、文本）以及发送消息的接口。
    此版本不再直接与 ChatEngine 的发送回调绑定，因为 ChatEngine 将直接处理通信。
    """

    def __init__(self,module_id, config: Dict[str, Any]):
        module_name = config.get("adapter_type")
        self.handler_config = config.get(module_name)
        logger.info(f"BaseHandler initialized for {self.__class__.__name__}")

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    def close(self):
        pass


