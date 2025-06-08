import contextvars
from typing import Optional, TYPE_CHECKING

# 避免循環導入的最佳實踐
if TYPE_CHECKING:
    from core.session_context import SessionContext

# 只保留一個 ContextVar，用於存儲完整的當前會話上下文對象
# 命名更清晰，表明它存儲的是 SessionContext
current_session_context_var: contextvars.ContextVar[Optional['SessionContext']] = \
    contextvars.ContextVar("current_session_context", default=None)
