from typing import Any, Coroutine, Optional, Type
from langchain_core.tools import BaseTool as LCBaseTool
from pydantic import BaseModel

class BaseTool(LCBaseTool):
    """Base class for all tools in this project, wrapping LangChain's BaseTool."""

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous execution of the tool."""
        raise NotImplementedError("Tool does not support synchronous execution.")

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous execution of the tool."""
        return self._run(*args, **kwargs)
