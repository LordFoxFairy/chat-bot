import logging
import functools
from typing import Type, Callable, Any, Optional

from backend.core.models.exceptions import ModuleProcessingError, FrameworkException

logger = logging.getLogger(__name__)

def handle_module_errors(error_class: Type[FrameworkException] = ModuleProcessingError, operation_name: str = "operation"):
    """
    自动捕获异常并转换为指定的模块异常类的装饰器

    Args:
        error_class: 要转换成的异常类，默认为ModuleProcessingError
        operation_name: 操作名称，用于日志记录
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except error_class:
                # 如果已经是目标异常类型，直接抛出
                raise
            except Exception as e:
                # 记录详细错误堆栈
                logger.error(f"Error during {operation_name}: {e}", exc_info=True)
                # 抛出转换后的异常，保留原始异常链
                raise error_class(f"Failed to perform {operation_name}: {str(e)}") from e
        return wrapper
    return decorator

def require_ready(func: Callable):
    """
    检查模块 is_ready 属性的装饰器
    如果未就绪，抛出 ModuleProcessingError
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_ready", False):
            raise ModuleProcessingError(f"Module {self.__class__.__name__} is not ready")
        return func(self, *args, **kwargs)
    return wrapper

def require_model(model_attr: str = "model"):
    """
    检查指定模型属性是否存在的装饰器
    如果属性为None，抛出 ModuleProcessingError

    Args:
        model_attr: 要检查的模型属性名称
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, model_attr, None) is None:
                raise ModuleProcessingError(f"Model attribute '{model_attr}' is not initialized in {self.__class__.__name__}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator
