import yaml
from typing import Dict, Any
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
