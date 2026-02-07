"""模块基类定义"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.utils.logging_setup import logger


# 配置模型类型变量
ConfigT = TypeVar("ConfigT", bound=BaseModel)


class BaseModule(ABC):
    """所有模块的基类

    提供模块生命周期管理和基础功能。

    职责:
    - 生命周期管理 (setup/close)
    - 状态管理 (is_ready)
    - 配置访问辅助
    - 配置验证支持

    生命周期:
        1. __init__: 读取配置，初始化变量
        2. setup(): 异步初始化资源（调用 _setup_impl）
        3. is_ready = True: 可以处理请求
        4. close(): 释放资源（调用 _close_impl）

    配置验证:
        子类可以通过 validate_config() 方法使用 Pydantic 模型验证配置：

        validated = self.validate_config(MyConfigModel)
        # validated 是经过验证的 Pydantic 模型实例
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        """初始化基础模块"""
        self.module_id = module_id
        self.config = config
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """模块是否准备好处理请求"""
        return self._is_ready

    @property
    def _log_prefix(self) -> str:
        """日志前缀，包含模块类名和 ID"""
        return f"{self.__class__.__name__} [{self.module_id}]"

    # ==================== 日志辅助方法 ====================

    def log_info(self, message: str) -> None:
        """记录 INFO 级别日志"""
        logger.info(f"{self._log_prefix} {message}")

    def log_debug(self, message: str) -> None:
        """记录 DEBUG 级别日志"""
        logger.debug(f"{self._log_prefix} {message}")

    def log_warning(self, message: str) -> None:
        """记录 WARNING 级别日志"""
        logger.warning(f"{self._log_prefix} {message}")

    def log_error(self, message: str, exc_info: bool = True) -> None:
        """记录 ERROR 级别日志

        Args:
            message: 日志消息
            exc_info: 是否包含异常堆栈信息（默认 True）
        """
        logger.error(f"{self._log_prefix} {message}", exc_info=exc_info)

    def log_critical(self, message: str, exc_info: bool = True) -> None:
        """记录 CRITICAL 级别日志"""
        logger.critical(f"{self._log_prefix} {message}", exc_info=exc_info)

    def session_log(
        self,
        level: str,
        session_id: str,
        message: str,
        exc_info: bool = False
    ) -> None:
        """记录带会话 ID 的日志

        Args:
            level: 日志级别 (info, debug, warning, error)
            session_id: 会话 ID
            message: 日志消息
            exc_info: 是否包含异常堆栈信息
        """
        full_message = f"{self._log_prefix} (Session: {session_id}) {message}"
        log_func = getattr(logger, level, logger.info)
        log_func(full_message, exc_info=exc_info)

    # ==================== 配置辅助方法 ====================

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键名
            default: 默认值

        Returns:
            配置值或默认值
        """
        return self.config.get(key, default)

    def require_config(self, key: str) -> Any:
        """获取必需的配置值

        Args:
            key: 配置键名

        Returns:
            配置值

        Raises:
            ValueError: 配置项不存在
        """
        value = self.config.get(key)
        if value is None:
            raise ValueError(f"缺少必需的配置项: {key}")
        return value

    def validate_config(self, config_model: Type[ConfigT]) -> ConfigT:
        """使用 Pydantic 模型验证配置

        Args:
            config_model: Pydantic 配置模型类

        Returns:
            验证后的配置模型实例

        Raises:
            ValueError: 配置验证失败

        Example:
            from src.core.config_models import LLMModuleConfig

            class MyLLMAdapter(BaseLLM):
                def __init__(self, module_id, config):
                    super().__init__(module_id, config)
                    validated = self.validate_config(LLMModuleConfig)
                    self.model_name = validated.model_name
                    self.temperature = validated.temperature
        """
        try:
            return config_model.model_validate(self.config)
        except ValidationError as e:
            raise ValueError(
                f"模块 {self.module_id} 配置验证失败: {e}"
            ) from e

    async def setup(self) -> None:
        """初始化模块资源（模板方法）"""
        if self._is_ready:
            return

        await self._setup_impl()
        self._is_ready = True

    @abstractmethod
    async def _setup_impl(self) -> None:
        """具体初始化逻辑（由子类实现）"""
        pass

    async def close(self) -> None:
        """关闭模块，释放资源（模板方法）"""
        if not self._is_ready:
            return

        await self._close_impl()
        self._is_ready = False

    async def _close_impl(self) -> None:
        """具体关闭逻辑（由子类覆盖）"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """健康检查

        Returns:
            健康状态信息
        """
        return {
            "module_id": self.module_id,
            "is_ready": self.is_ready,
            "status": "healthy" if self.is_ready else "not_ready",
        }

    async def __aenter__(self) -> "BaseModule":
        """异步上下文管理器入口"""
        await self.setup()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.close()
