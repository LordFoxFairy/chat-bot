"""核心异常定义"""

from typing import Any, Dict, Optional


class FrameworkException(Exception):
    """框架基础异常"""

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details,
        }


class ModuleInitializationError(FrameworkException):
    """模块初始化错误"""

    def __init__(
        self,
        message: str,
        *,
        module_id: Optional[str] = None,
        adapter_type: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if module_id:
            self.details["module_id"] = module_id
        if adapter_type:
            self.details["adapter_type"] = adapter_type


class ModuleProcessingError(FrameworkException):
    """模块处理错误"""

    pass


class PipelineExecutionError(FrameworkException):
    """管道执行错误"""

    pass


class ConfigurationError(FrameworkException):
    """配置错误"""

    def __init__(
        self,
        message: str,
        *,
        config_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if config_key:
            self.details["config_key"] = config_key
