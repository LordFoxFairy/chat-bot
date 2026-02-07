from typing import Any, Dict
from utils.logging_setup import logger

class InterruptManager:
    """管理对话打断状态

    职责:
    - 维护当前是否处于打断状态 (interrupt_flag)
    - 维护本轮对话是否发生过打断 (was_interrupted)
    - 提供状态查询和变更方法
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._interrupt_flag = False
        self._was_interrupted = False

    @property
    def is_interrupted(self) -> bool:
        """当前是否处于打断状态"""
        return self._interrupt_flag

    @property
    def was_interrupted(self) -> bool:
        """本轮对话是否发生过打断（用于上下文拼接）"""
        return self._was_interrupted

    def set_interrupt(self):
        """设置打断状态"""
        if not self._interrupt_flag:
            self._interrupt_flag = True
            self._was_interrupted = True
            logger.debug(f"ConversationHandler 检测到打断: session={self.session_id}")

    def reset(self):
        """重置当前打断标志（通常在处理完打断后调用）"""
        self._interrupt_flag = False

    def reset_history(self):
        """重置历史打断记录"""
        self._was_interrupted = False

    def check_interrupt(self) -> bool:
        """检查并返回是否被打断"""
        return self._interrupt_flag
