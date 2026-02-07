import time
from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from src.utils.logging_setup import logger

@wrap_model_call
def retry_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """
    Middleware that retries the model call if it fails.
    Defaults to 3 attempts with exponential backoff.
    """
    max_retries = 3
    base_delay = 1.0  # seconds

    for attempt in range(max_retries):
        try:
            return handler(request)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Model call failed after {max_retries} attempts: {e}")
                raise

            delay = base_delay * (2 ** attempt)
            logger.warning(
                f"Model call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)

    # Should be unreachable due to the raise in the loop, but for type safety
    raise RuntimeError("Unexpected retry loop exit")
