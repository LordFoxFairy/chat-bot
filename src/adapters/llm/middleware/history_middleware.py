import json
from typing import Callable, Any, Dict, Optional
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.agents.middleware import AgentState, AgentMiddleware, wrap_model_call
from src.utils.logging_setup import logger

class HistoryMiddleware(AgentMiddleware):
    def __init__(self, history_file: str = "history.json"):
        self.history_file = history_file
        self._cache = []

    def load_history(self) -> list[BaseMessage]:
        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
                return [self._parse_message(msg) for msg in data]
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing history found, starting fresh.")
            return []

    def save_history(self, messages: list[BaseMessage]):
        serialized = [self._serialize_message(msg) for msg in messages]
        with open(self.history_file, "w") as f:
            json.dump(serialized, f, indent=2)

    def _serialize_message(self, message: BaseMessage) -> dict:
        return {
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
        }

    def _parse_message(self, data: dict) -> BaseMessage:
        msg_type = data.get("type")
        content = data.get("content", "")
        kwargs = data.get("additional_kwargs", {})

        if msg_type == "human":
            return HumanMessage(content=content, additional_kwargs=kwargs)
        elif msg_type == "ai":
            return AIMessage(content=content, additional_kwargs=kwargs)
        elif msg_type == "system":
            return SystemMessage(content=content, additional_kwargs=kwargs)
        else:
            return HumanMessage(content=content, additional_kwargs=kwargs)

    @wrap_model_call
    def wrap_model_call(
        self,
        request: AgentState,
        handler: Callable[[AgentState], Any],
    ) -> Any:
        # Load history before model call
        # In a real agent these would be merged into the state
        # For this middleware example, we just log and persist
        try:
            history = self.load_history()
            logger.info(f"Loaded {len(history)} messages from history")

            # Execute the model call
            response = handler(request)

            # Persist the updated state (if available) or result
            # Assuming 'messages' key in state or result holds the conversation
            messages = []
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
            elif isinstance(request, dict) and "messages" in request:
                messages = request["messages"]

            if messages:
                self.save_history(messages)
                logger.debug(f"Saved {len(messages)} messages to history")

            return response

        except Exception as e:
            logger.error(f"Error in history middleware: {e}")
            raise
