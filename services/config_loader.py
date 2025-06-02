import yaml
from typing import Dict, Any
from core.exceptions import ConfigurationError

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            print(f"配置加载器: 成功从 '{config_path}' 加载配置。")
            return config_data
        except FileNotFoundError:
            msg = f"配置文件 '{config_path}' 未找到。"
            print(f"配置加载器: 错误 - {msg}")
            raise ConfigurationError(msg)
        except yaml.YAMLError as e:
            msg = f"解析配置文件 '{config_path}' 失败: {e}"
            print(f"配置加载器: 错误 - {msg}")
            raise ConfigurationError(msg) from e
        except Exception as e:
            msg = f"加载配置文件时发生未知错误: {e}"
            print(f"配置加载器: {msg}")
            raise ConfigurationError(msg) from e