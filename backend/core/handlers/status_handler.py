from typing import Any, Callable, Dict, Optional

class StatusHandler:
    """处理系统状态获取逻辑"""

    def __init__(self, module_provider: Callable[[str], Optional[Any]]):
        """初始化

        Args:
            module_provider: 用于获取模块实例的 Callable
        """
        self.module_provider = module_provider

    async def handle_status_get(self) -> Dict[str, Any]:
        """获取所有模块的状态报告

        Returns:
            Dict[str, Any]: 包含各模块状态的字典，键为模块类型，值为状态信息
        """
        # 收集所有模块状态
        status_report = {}

        # 尝试获取几个核心模块的状态
        modules_to_check = ['asr', 'vad', 'llm', 'tts']

        for module_type in modules_to_check:
            module = self.module_provider(module_type)
            if module:
                status_report[module_type] = {
                    "status": "running",
                    "module_id": getattr(module, 'module_id', 'unknown'),
                    "module_type": module.__class__.__name__,
                    "initialized": getattr(module, '_initialized', False),
                }
            else:
                status_report[module_type] = {
                    "status": "stopped",
                    "error": "Not loaded"
                }

        return status_report
