"""配置管理器

提供配置的读取、更新和敏感信息掩码功能。
"""

import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
import aiofiles

from backend.core.models.config_data import ConfigData
from backend.core.models.exceptions import ConfigurationError
from backend.utils.logging_setup import logger


# 敏感字段关键词模式
SENSITIVE_FIELD_PATTERNS: List[str] = [
    r".*api[_-]?key.*",
    r".*secret.*",
    r".*password.*",
    r".*token.*",
    r".*credential.*",
    r".*auth.*",
    r".*private[_-]?key.*",
]

# 编译正则表达式
_SENSITIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_FIELD_PATTERNS]

# 掩码占位符
MASK_PLACEHOLDER = "******"


class ConfigManager:
    """配置管理器

    提供配置的读取、更新和敏感信息掩码功能。
    支持按 section 获取/更新配置。
    """

    def __init__(self, config_path: str):
        """初始化配置管理器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config_cache: Optional[Dict[str, Any]] = None

    async def get_config(
        self,
        section: Optional[str] = None,
        mask_sensitive: bool = True
    ) -> ConfigData:
        """获取配置

        Args:
            section: 配置区段名称，None 表示获取全部配置
            mask_sensitive: 是否掩码敏感字段

        Returns:
            ConfigData 对象

        Raises:
            ConfigurationError: 配置读取失败
        """
        try:
            config = await self._load_config()

            if section:
                # 获取指定区段
                if section not in config:
                    raise ConfigurationError(f"配置区段 '{section}' 不存在")
                content = {section: config[section]}
            else:
                content = config

            # 掩码敏感信息
            if mask_sensitive:
                content = mask_sensitive_fields(content)

            return ConfigData(section=section, content=content)

        except ConfigurationError:
            raise
        except Exception as e:
            msg = f"获取配置失败: {e}"
            logger.error(f"ConfigManager: {msg}", exc_info=True)
            raise ConfigurationError(msg) from e

    async def update_config(
        self,
        updates: Dict[str, Any],
        section: Optional[str] = None,
        validate: bool = True
    ) -> ConfigData:
        """更新配置

        Args:
            updates: 要更新的配置内容
            section: 配置区段名称，None 表示更新整个配置
            validate: 是否验证配置

        Returns:
            更新后的 ConfigData 对象

        Raises:
            ConfigurationError: 配置更新失败
        """
        try:
            config = await self._load_config()

            if section:
                # 更新指定区段
                if section not in config:
                    raise ConfigurationError(f"配置区段 '{section}' 不存在")

                # 深度合并更新
                config[section] = self._deep_merge(config[section], updates)
            else:
                # 更新整个配置
                config = self._deep_merge(config, updates)

            # 验证配置
            if validate:
                self._validate_config(config)

            # 保存配置
            await self._save_config(config)

            # 清除缓存
            self._config_cache = None

            # 返回更新后的配置（掩码敏感信息）
            return await self.get_config(section=section, mask_sensitive=True)

        except ConfigurationError:
            raise
        except Exception as e:
            msg = f"更新配置失败: {e}"
            logger.error(f"ConfigManager: {msg}", exc_info=True)
            raise ConfigurationError(msg) from e

    async def _load_config(self) -> Dict[str, Any]:
        """加载配置文件

        Returns:
            配置字典
        """
        if self._config_cache is not None:
            return copy.deepcopy(self._config_cache)

        try:
            async with aiofiles.open(self.config_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                config = yaml.safe_load(content) or {}
                self._config_cache = config
                return copy.deepcopy(config)
        except FileNotFoundError:
            raise ConfigurationError(f"配置文件 '{self.config_path}' 不存在")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"配置文件格式错误: {e}")

    async def _save_config(self, config: Dict[str, Any]) -> None:
        """保存配置到文件

        Args:
            config: 配置字典
        """
        try:
            async with aiofiles.open(self.config_path, 'w', encoding='utf-8') as f:
                content = yaml.dump(
                    config,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False
                )
                await f.write(content)
            logger.info(f"ConfigManager: 配置已保存到 '{self.config_path}'")
        except Exception as e:
            raise ConfigurationError(f"保存配置失败: {e}")

    def _deep_merge(
        self,
        base: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """深度合并两个字典

        Args:
            base: 基础字典
            updates: 更新字典

        Returns:
            合并后的字典
        """
        result = copy.deepcopy(base)

        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置

        Args:
            config: 配置字典

        Raises:
            ConfigurationError: 配置验证失败
        """
        # 基本验证：确保必要的顶级字段存在
        required_sections = ["modules"]

        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"缺少必要的配置区段: '{section}'")

        # 验证 modules 配置
        modules = config.get("modules", {})
        if not isinstance(modules, dict):
            raise ConfigurationError("'modules' 必须是一个字典")

        # 可以添加更多具体的验证逻辑

    def invalidate_cache(self) -> None:
        """使缓存失效"""
        self._config_cache = None


def is_sensitive_field(field_name: str) -> bool:
    """判断字段名是否为敏感字段

    Args:
        field_name: 字段名

    Returns:
        是否为敏感字段
    """
    for pattern in _SENSITIVE_PATTERNS:
        if pattern.match(field_name):
            return True
    return False


def mask_sensitive_fields(
    data: Any,
    parent_key: str = ""
) -> Any:
    """递归掩码敏感字段

    Args:
        data: 要处理的数据
        parent_key: 父级键名（用于上下文判断）

    Returns:
        处理后的数据
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if is_sensitive_field(key):
                # 敏感字段，进行掩码
                if isinstance(value, str) and value:
                    result[key] = MASK_PLACEHOLDER
                elif isinstance(value, dict):
                    # 如果值是字典，整个替换为掩码
                    result[key] = MASK_PLACEHOLDER
                else:
                    result[key] = MASK_PLACEHOLDER
            else:
                # 非敏感字段，递归处理
                result[key] = mask_sensitive_fields(value, key)
        return result
    elif isinstance(data, list):
        return [mask_sensitive_fields(item, parent_key) for item in data]
    else:
        return data


def unmask_sensitive_fields(
    masked_config: Dict[str, Any],
    original_config: Dict[str, Any]
) -> Dict[str, Any]:
    """将掩码的敏感字段还原为原始值

    用于更新配置时，如果用户没有修改敏感字段（仍为掩码值），
    则保留原始值。

    Args:
        masked_config: 可能包含掩码的配置
        original_config: 原始配置

    Returns:
        还原后的配置
    """
    if not isinstance(masked_config, dict) or not isinstance(original_config, dict):
        return masked_config

    result = {}
    for key, value in masked_config.items():
        if key in original_config:
            if value == MASK_PLACEHOLDER:
                # 保留原始值
                result[key] = original_config[key]
            elif isinstance(value, dict) and isinstance(original_config.get(key), dict):
                # 递归处理嵌套字典
                result[key] = unmask_sensitive_fields(value, original_config[key])
            else:
                result[key] = value
        else:
            result[key] = value

    return result


# 全局配置管理器实例（延迟初始化）
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """获取配置管理器实例

    Args:
        config_path: 配置文件路径，首次调用时必须提供

    Returns:
        ConfigManager 实例
    """
    global _config_manager

    if _config_manager is None:
        if config_path is None:
            # 使用默认路径
            config_path = "backend/configs/config.yaml"
        _config_manager = ConfigManager(config_path)

    return _config_manager
