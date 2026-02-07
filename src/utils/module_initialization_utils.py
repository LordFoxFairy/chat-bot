from typing import Any, Dict, Optional, Callable, Union, Type

from src.core.models.exceptions import ModuleInitializationError
from src.core.interfaces.base_protocol import BaseProtocol
from src.core.interfaces.base_module import BaseModule
from src.utils.logging_setup import logger


def resolve_adapter_config(module_config: Dict[str, Any]) -> Dict[str, Any]:
    """解析适配器配置，支持 enable_module 选择子配置

    如果配置中包含 enable_module 字段，则从 config 子字典中选取对应的配置。
    否则直接返回整个 config 字典。

    同时将顶层配置（如 system_prompt）合并到最终配置中。

    Args:
        module_config: 模块配置字典

    Returns:
        解析后的适配器配置字典
    """
    base_config = module_config.get("config", {})
    enable_module = module_config.get("enable_module")

    if enable_module and isinstance(base_config, dict):
        # 从 config 中选取 enable_module 指定的子配置
        selected_config = base_config.get(enable_module, {})

        if not selected_config:
            logger.warning(
                f"enable_module='{enable_module}' 指定的配置不存在于 config 中，"
                f"可用选项: {list(base_config.keys())}"
            )
            selected_config = {}

        # 合并顶层配置（如 system_prompt）到选中的配置中
        top_level_keys = ["system_prompt", "max_tokens", "temperature"]
        for key in top_level_keys:
            if key in module_config and key not in selected_config:
                selected_config[key] = module_config[key]

        logger.debug(f"使用 enable_module='{enable_module}' 选择配置")
        return selected_config

    # 如果没有 enable_module，尝试使用 adapter_type 作为 key
    adapter_type = module_config.get("adapter_type")
    if adapter_type and isinstance(base_config, dict) and adapter_type in base_config:
        return base_config.get(adapter_type, {})

    return base_config


async def initialize_single_module_instance(
        module_id: str,
        module_config: dict,
        factory_dict: Dict[str, Callable[..., Union[BaseModule, BaseProtocol]]],
        base_class: Union[Type[BaseModule], Type[BaseProtocol]],
        existing_modules: Dict[str, Union[BaseModule, BaseProtocol]],
        conversation_manager: Optional['ConversationManager'] = None
) -> Optional[Union[BaseModule, BaseProtocol]]:
    """
    通用函数，用于初始化单个模块实例。
    处理配置检查、工厂查找、实例创建、类型验证和错误捕获。

    Args:
        module_id (str): 模块的唯一标识符。
        module_config (dict): 当前模块的配置。
        factory_dict (Dict[str, Callable]): 模块工厂函数的字典，用于创建模块实例。
        base_class (Union[Type[BaseModule], Type[BaseProtocol]]): 模块实例应继承的基类，用于类型检查。
        existing_modules (Dict[str, Union[BaseModule, BaseProtocol]]): 已加载模块的字典，用于检查重复。
        conversation_manager (Optional[ConversationManager]): ConversationManager 实例，用于协议适配器会话管理。

    Returns:
        Optional[Union[BaseModule, BaseProtocol]]: 初始化成功的模块实例，失败则返回 None。
    """
    if not isinstance(module_config, dict):
        logger.error("模块 '%s' 的配置必须是 dict，实际为: %s", module_id, type(module_config).__name__)
        return None

    if module_id in existing_modules:
        logger.warning("模块 '%s' 已存在，跳过初始化。", module_id)
        return None

    factory = factory_dict.get(module_id)
    if not factory:
        logger.error("未注册模块类型 '%s' 的创建工厂函数。", module_id)
        return None

    try:
        logger.info("初始化模块 '%s'...", module_id)
        # 调用工厂函数创建模块实例
        adapter_type = module_config.get("adapter_type")

        # 解析适配器配置（支持 enable_module 选择子配置）
        adapter_config = resolve_adapter_config(module_config)

        if conversation_manager is not None:
            module_instance: Union[BaseModule, BaseProtocol] = factory(
                adapter_type=adapter_type,
                module_id=module_id,
                config=adapter_config,
                conversation_manager=conversation_manager
            )
        else:
            module_instance: Union[BaseModule, BaseProtocol] = factory(
                adapter_type=adapter_type,
                module_id=module_id,
                config=adapter_config
            )

        # 验证模块实例的类型
        if not isinstance(module_instance, base_class):
            raise ModuleInitializationError(
                f"模块 '{module_id}' 实例不是 {base_class.__name__} 的子类，实际类型: {type(module_instance).__name__}"
            )

        # 初始化模块并添加到已加载模块字典
        await module_instance.setup()
        existing_modules[module_id] = module_instance
        logger.info("模块 '%s' 初始化成功。", module_id)
        return module_instance

    except ModuleInitializationError as e:
        logger.error("模块 '%s' 初始化失败: %s", module_id, e)
    except Exception as e:
        logger.exception("模块 '%s' 初始化时发生异常: %s", module_id, e)
    return None
