import json
import logging
import base64
from typing import Any, Dict, TYPE_CHECKING, Union, Optional

from data_models.audio_data import AudioData  # 使用更新后的 AudioData
from data_models.text_data import TextData

# 使用TYPE_CHECKING来避免循环导入，仅用于类型提示
if TYPE_CHECKING:
    from core.chat_engine import ChatEngine  # 引用重构后的 ChatEngine

logger = logging.getLogger(__name__)


class WebsocketInputHandler:
    """
    处理来自WebSocket的原始消息，将其解析、预处理，
    然后传递给ChatEngine，并返回ChatEngine的处理结果。
    """

    def __init__(self, chat_engine: 'ChatEngine'):
        self.chat_engine = chat_engine

    async def handle_message(self, session_id: str, raw_message: str) -> Dict[str, Any]:
        """
        处理从WebSocket接收到的单个原始消息字符串。

        :param session_id: 当前会话ID。
        :param raw_message: 从WebSocket接收到的原始JSON字符串消息。
        :return: 一个包含处理结果的字典，预备发送回客户端。
        """
        client_message_id: Optional[str] = None
        parsed_data: Optional[Dict[str, Any]] = None

        try:
            parsed_data = json.loads(raw_message)
            message_type = parsed_data.get("type")
            payload = parsed_data.get("payload")
            client_message_id = parsed_data.get("message_id", f"ws_msg_{session_id}")

            if not message_type or payload is None:
                logger.warning(f"会话 {session_id}: 无效的消息格式，缺少'type'或'payload'。消息: {raw_message[:200]}")
                return self._format_error_response(client_message_id, session_id,
                                                   "无效的消息格式，缺少'type'或'payload'字段。")

            engine_input_payload: Union[AudioData, TextData, str, None] = None
            input_origin_type: str = ""  # "audio" 或 "text"，告知ChatEngine原始输入类型

            if message_type == "audio_input":
                if not isinstance(payload, dict) or "data" not in payload:
                    logger.warning(f"会话 {session_id}: 无效的audio_input payload: {payload}")
                    return self._format_error_response(client_message_id, session_id, "音频输入payload格式错误。")
                try:
                    audio_b64_data = payload.get("data")
                    # 客户端应明确告知音频类型，默认为 "audio/pcm"
                    # 如果客户端发送的是如 "audio/wav" 这样的封装格式，ASR模块需要能处理它
                    audio_content_type = payload.get("content_type", "audio/pcm").lower()

                    if not isinstance(audio_b64_data, str):
                        return self._format_error_response(client_message_id, session_id,
                                                           "音频数据必须是Base64编码的字符串。")

                    audio_bytes = base64.b64decode(audio_b64_data)

                    # 对于PCM，这些参数是必需的。对于其他类型，它们可能嵌入在数据中或由ASR模块确定。
                    sample_rate = int(payload.get("sample_rate", 16000))
                    channels = int(payload.get("channels", 1))
                    sample_width = int(payload.get("sample_width", 2))  # 每个样本的字节数 (例如 2 for 16-bit)

                    engine_input_payload = AudioData(
                        audio=audio_bytes,
                        type=audio_content_type,  # 使用客户端提供的类型
                        sample_rate=sample_rate,
                        channels=channels,
                        sample_width=sample_width
                        # metadata={"session_id": session_id} # 如果AudioData需要携带会话ID
                    )
                    input_origin_type = "audio"
                except (base64.binascii.Error, ValueError, TypeError) as e:
                    logger.error(f"会话 {session_id}: 处理audio_input payload时出错: {e}", exc_info=True)
                    return self._format_error_response(client_message_id, session_id,
                                                       f"音频数据解码或参数处理失败: {e}")

            elif message_type == "text_input":
                if not isinstance(payload, dict) or "text" not in payload:
                    logger.warning(f"会话 {session_id}: 无效的text_input payload: {payload}")
                    return self._format_error_response(client_message_id, session_id, "文本输入payload格式错误。")

                text_content = payload.get("text")
                if not isinstance(text_content, str):
                    return self._format_error_response(client_message_id, session_id, "文本内容必须是字符串。")

                # ChatEngine的process_message可以直接接收str或TextData
                engine_input_payload = text_content
                input_origin_type = "text"

            else:
                logger.warning(f"会话 {session_id}: 不支持的消息类型 '{message_type}'")
                return self._format_error_response(client_message_id, session_id, f"不支持的消息类型: {message_type}")

            if engine_input_payload is None or not input_origin_type:
                logger.error(f"会话 {session_id}: 无法为引擎准备有效的输入数据，消息类型: {message_type}")
                return self._format_error_response(client_message_id, session_id, "无法准备有效的引擎输入数据。")

            # 调用ChatEngine进行核心处理
            engine_response = await self.chat_engine.process_message(
                session_id=session_id,
                input_payload=engine_input_payload,  # 传递 AudioData, TextData 或 str
                input_origin_type=input_origin_type  # 告知 ChatEngine 原始输入类型
            )

            # 将ChatEngine的响应包装成最终发送给客户端的格式
            final_response = {
                "type": "bot_response",
                "session_id": engine_response.get("session_id", session_id),
                "text": engine_response.get("text"),
                "audio_b64": engine_response.get("audio_b64"),
                "error": engine_response.get("error"),
                "original_message_id": client_message_id
            }
            return final_response

        except json.JSONDecodeError:
            logger.error(f"会话 {session_id}: 解码JSON失败: {raw_message[:200]}", exc_info=True)
            return self._format_error_response(client_message_id or f"json_err_{session_id}", session_id,
                                               "无效的JSON格式。")
        except Exception as e:
            logger.error(f"WebsocketInputHandler在处理会话 {session_id} 时发生意外错误: {e}", exc_info=True)
            return self._format_error_response(client_message_id or f"handler_err_{session_id}", session_id,
                                               f"服务器处理消息时发生意外错误: {str(e)}")

    def _format_error_response(self, client_message_id: Optional[str], session_id: str, error_message: str) -> Dict[
        str, Any]:
        """辅助方法，用于格式化错误响应。"""
        return {
            "type": "error_response",
            "session_id": session_id,
            "error": error_message,
            "original_message_id": client_message_id
        }
