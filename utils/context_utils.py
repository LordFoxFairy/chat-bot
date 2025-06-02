# core_framework/context_utils.py
import contextvars
from typing import Optional, TYPE_CHECKING

# 仅在类型检查时导入 SessionContext，以避免运行时循环导入
# 这是因为 SessionContext 可能会在其日志或其他地方间接使用到 context_utils (虽然不常见)
# 或者其他导入了 context_utils 的模块也被 SessionContext 导入。
# 使用字符串类型提示是更安全的做法。
if TYPE_CHECKING:
    from core.session_context import SessionContext # 假设 SessionContext 在同级目录

# 用于存储当前会话ID的上下文变量 (可选保留，如果某些地方仍需直接访问ID)
current_session_id_var: contextvars.ContextVar[Optional[str]] = \
    contextvars.ContextVar("current_session_id", default=None)

# 用于直接存储当前 SessionContext 对象的上下文变量
# 类型提示使用字符串 'SessionContext' 来避免直接的顶层导入，防止潜在的循环依赖。
# Python 的类型检查器 (如 MyPy) 和许多 IDE 能够正确处理这种前向引用。
current_user_session_var: contextvars.ContextVar[Optional['SessionContext']] = \
    contextvars.ContextVar("current_user_session_context", default=None) # 变量名更清晰

# 使用示例 (在其他模块中):
# from core_framework.context_utils import current_user_session_var
#
# async def some_function():
#     current_session = current_user_session_var.get()
#     if current_session:
#         # 使用 current_session.session_id, current_session.session_config 等
#         logger.info(f"当前处理的会话ID: {current_session.session_id}")
#     else:
#         logger.warning("在当前上下文中未找到用户会话。")
