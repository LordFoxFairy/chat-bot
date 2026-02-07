import json
from typing import Any, Dict, Optional
from langchain.agents.middleware import (
    AgentSensitive,
    before_model,
    after_model,
    AgentState,
    Runtime,
)
from src.utils.logging_setup import logger

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> AgentSensitive | Dict[str, Any] | None:
    """
    Middleware that logs metadata and messages before a model call.
    """
    num_messages = len(state.get("messages", []))

    logger.info(f"Preparing to call model with {num_messages} messages")

    # Log specific state keys if present, avoiding sensitive data if marked
    log_data = {
        "messages_count": num_messages,
        "runtime_config": runtime.config if hasattr(runtime, "config") else {},
    }

    logger.debug(f"Pre-model context: {json.dumps(log_data, default=str)}")

    return None

@after_model
def log_after_model(state: AgentState, result: Any, runtime: Runtime) -> AgentSensitive | Dict[str, Any] | None:
    """
    Middleware that logs the result after a model call.
    """
    logger.info("Model call completed successfully")

    # Safely log result summary
    if isinstance(result, dict):
        # If result is a dict (common in LangGraph based agents)
        logger.debug(f"Model result keys: {list(result.keys())}")
        if "messages" in result:
             logger.debug(f"Model generated {len(result['messages'])} new messages")
    else:
        # Fallback for other return types
        logger.debug(f"Model result type: {type(result).__name__}")

    return None
