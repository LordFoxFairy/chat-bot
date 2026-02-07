import os
import re
import yaml
from typing import Any, Dict
import aiofiles
from core.exceptions import ConfigurationError
from utils.logging_setup import logger


class ConfigLoader:
    @staticmethod
    async def load_config(config_path: str) -> Dict[str, Any]:
        """异步加载 YAML 配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典

        Raises:
            ConfigurationError: 配置加载失败
        """
        try:
            async with aiofiles.open(config_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                config_data = yaml.safe_load(content)
            logger.info(f"配置加载器: 成功从 '{config_path}' 加载配置。")
            return config_data
        except FileNotFoundError:
            msg = f"配置文件 '{config_path}' 未找到。"
            logger.error(f"配置加载器: 错误 - {msg}")
            raise ConfigurationError(msg)
        except yaml.YAMLError as e:
            msg = f"解析配置文件 '{config_path}' 失败: {e}"
            logger.error(f"配置加载器: 错误 - {msg}")
            raise ConfigurationError(msg) from e
        except Exception as e:
            msg = f"加载配置文件时发生未知错误: {e}"
            logger.error(f"配置加载器: {msg}")
            raise ConfigurationError(msg) from e

    @staticmethod
    def resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """递归解析配置中的环境变量引用

        支持格式:
        - ${ENV_VAR}: 必须存在的环境变量
        - ${ENV_VAR:default}: 带默认值的环境变量

        Args:
            config: 配置字典

        Returns:
            解析后的配置字典
        """
        env_pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                def replacer(match: re.Match) -> str:
                    env_name = match.group(1)
                    default = match.group(2)
                    env_value = os.getenv(env_name)
                    if env_value is not None:
                        return env_value
                    if default is not None:
                        return default
                    logger.warning(f"环境变量 '{env_name}' 未设置且无默认值")
                    return match.group(0)  # 保持原样

                return env_pattern.sub(replacer, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        return resolve_value(config)
