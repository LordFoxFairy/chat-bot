from typing import Any, Dict, Optional
import logging

from backend.utils.config_manager import get_config_manager, unmask_sensitive_fields

logger = logging.getLogger(__name__)

class ConfigHandler:
    """配置处理类

    职责:
    - 处理配置的获取请求
    - 处理配置的更新请求
    - 独立于具体的通信协议
    """

    def __init__(self, config_path: str = "backend/configs/config.yaml"):
        self.config_path = config_path

    async def handle_config_get(self, section: Optional[str] = None) -> Dict[str, Any]:
        """获取配置

        Args:
            section: 配置部分名称，如果为None则获取全部配置

        Returns:
            Dict[str, Any]: 配置内容字典

        Raises:
            Exception: 获取配置失败时抛出异常
        """
        try:
            # 使用配置管理器获取配置（自动掩码敏感信息）
            config_manager = get_config_manager(self.config_path)
            config_data = await config_manager.get_config(
                section=section,
                mask_sensitive=True
            )
            return config_data.content

        except Exception as e:
            logger.error(f"Failed to get config: {e}", exc_info=True)
            raise

    async def handle_config_set(self, new_config: Dict[str, Any], section: Optional[str] = None) -> Dict[str, Any]:
        """更新配置

        Args:
            new_config: 新的配置字典
            section: 配置部分名称，如果为None则更新全部或根据new_config中的_section字段

        Returns:
            Dict[str, Any]: 更新后的配置内容字典

        Raises:
            ValueError: 配置格式无效
            Exception: 更新配置失败
        """
        try:
            if not isinstance(new_config, dict):
                raise ValueError("Valid configuration dictionary required")

            # 如果未指定section，尝试从配置中获取
            if section is None:
                section = new_config.pop("_section", None)

            # 使用配置管理器更新配置
            config_manager = get_config_manager(self.config_path)

            # 获取原始配置（用于还原掩码的敏感字段）
            original_config = await config_manager.get_config(
                section=section,
                mask_sensitive=False
            )

            # 还原掩码的敏感字段（如果用户没有修改）
            # 注意：unmask_sensitive_fields 需要原始配置的内容
            config_to_save = unmask_sensitive_fields(new_config, original_config.content)

            # 更新配置
            updated_config = await config_manager.update_config(
                updates=config_to_save,
                section=section,
                validate=True
            )

            logger.info("Configuration updated successfully")

            # 返回更新后的配置（带掩码）
            return updated_config.content

        except Exception as e:
            logger.error(f"Failed to set config: {e}", exc_info=True)
            raise
