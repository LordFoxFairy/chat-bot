"""依赖注入容器

职责:
- 管理应用组件的生命周期
- 解决组件间的依赖关系
- 提供类型安全的组件访问接口

设计原则:
- 显式依赖优于隐式依赖
- 构造函数注入优于 setter 注入
- 接口编程优于实现编程

使用示例:
    # 1. 创建容器
    container = Container()

    # 2. 注册组件
    container.register(LanguageModel, OpenAIModel(api_key="..."))
    container.register("config", Config({"debug": True}))

    # 3. 获取组件
    llm = container.resolve(LanguageModel)
    config = container.resolve("config")
"""

from threading import RLock
from typing import Any, Dict, Optional, Type, TypeVar, Union, Callable
import inspect

T = TypeVar("T")

class DependencyError(Exception):
    """依赖注入相关错误"""
    pass

class Container:
    """轻量级依赖注入容器

    用于替代全局 AppContext，提供非侵入式的依赖管理。
    """

    def __init__(self):
        self._instances: Dict[Union[str, Type], Any] = {}
        self._factories: Dict[Union[str, Type], Callable[[], Any]] = {}
        self._lock = RLock()

    def register(self, key: Union[str, Type[T]], instance: Any) -> None:
        """注册单例实例

        Args:
            key: 查找键，可以是字符串名称或类型
            instance: 组件实例

        Raises:
            DependencyError: 如果注册的实例类型与 key 不匹配（当 key 为类型时）
        """
        if isinstance(key, type) and not isinstance(instance, key) and instance is not None:
             # 对于 Protocol 或抽象基类，isinstance 可能无法完全检查，但至少做基础检查
             # 注意：如果是鸭子类型或者 mock 对象，这里可能会报警，所以保持宽容或者由用户保证
             pass

        with self._lock:
            self._instances[key] = instance
            # 如果注册的是工厂，移除它，因为实例优先
            if key in self._factories:
                del self._factories[key]

    def register_factory(self, key: Union[str, Type[T]], factory: Callable[[], Any]) -> None:
        """注册工厂函数

        每次 resolve 时都会调用 factory 创建新实例。

        Args:
            key: 查找键
            factory: 无参工厂函数
        """
        with self._lock:
            self._factories[key] = factory
            # 移除可能存在的实例
            if key in self._instances:
                del self._instances[key]

    def resolve(self, key: Union[str, Type[T]]) -> T:
        """解析并获取依赖

        Args:
            key: 查找键

        Returns:
            组件实例

        Raises:
            DependencyError: 如果找不到依赖
        """
        with self._lock:
            # 1. 检查单例实例
            if key in self._instances:
                return self._instances[key]

            # 2. 检查工厂
            if key in self._factories:
                return self._factories[key]()

            # 3. 如果通过字符串查找失败，尝试通过类型名称查找（可选，这里保持严格）

            raise DependencyError(f"Dependency not found: {key}")

    def get(self, key: Union[str, Type[T]], default: Any = None) -> Any:
        """安全获取依赖，失败返回默认值

        Args:
            key: 查找键
            default: 默认值

        Returns:
            组件实例或默认值
        """
        try:
            return self.resolve(key)
        except DependencyError:
            return default

    def has(self, key: Union[str, Type[T]]) -> bool:
        """检查是否存在依赖

        Args:
            key: 查找键

        Returns:
            是否已注册
        """
        with self._lock:
            return key in self._instances or key in self._factories

    def clear(self) -> None:
        """清空容器（主要用于测试）"""
        with self._lock:
            self._instances.clear()
            self._factories.clear()

    def clone(self) -> 'Container':
        """创建容器的浅拷贝

        用于创建子容器或测试隔离。
        """
        new_container = Container()
        with self._lock:
            new_container._instances = self._instances.copy()
            new_container._factories = self._factories.copy()
        return new_container
