import asyncio
import base64
import json
import re
import uuid
from typing import Optional, List, Dict

import websockets

import constant
from core.session_context import SessionContext
from core.session_manager import session_manager
from data_models import StreamEvent, EventType, TextData
from modules import BaseLLM, BaseTTS, BaseVAD, BaseASR
from modules.base_protocol import BaseProtocol
from service.AudioConsumer import AudioConsumer
from utils.logging_setup import logger


class WebSocketProtocolAdapter(BaseProtocol):
    def __init__(self, module_id, config):
        super().__init__(module_id, config)
        self.host = self.handler_config.get("host")
        self.port = self.handler_config.get("port")
        self.server = None
        self.tag_to_current_session = {}
        self.session_to_websocket = {}
        self.websocket_to_session_id = {}
        self.audio_consumers: Dict[str, AudioConsumer] = {}
        self.session_interrupt_flags: Dict[str, bool] = {}

        # 新增：用于管理每个会话的对话轮次上下文
        self.session_turn_context: Dict[str, Dict] = {}

        print(f"[服务器] WebSocketProtocolAdapter 已初始化，监听地址: {self.host}:{self.port}")

    async def register(self, websocket):
        register_data = await websocket.recv()
        stream_event = StreamEvent.model_validate_json(register_data)
        if stream_event.event_type == EventType.SYSTEM_CLIENT_SESSION_START:
            client_tag_id = stream_event.tag_id
            current_tag_id = client_tag_id
            current_session_id = str(uuid.uuid4())

            self.tag_to_current_session[current_tag_id] = current_session_id
            self.session_to_websocket[current_session_id] = websocket
            self.websocket_to_session_id[websocket] = current_session_id

            # 新增：为新会话初始化轮次上下文
            self.session_turn_context[current_session_id] = {
                'last_user_text': '',  # 上一轮用户说的完整文本
                'was_interrupted': False  # 本轮是否是一次打断
            }

            assignment_message = StreamEvent(
                event_type=EventType.SYSTEM_SERVER_SESSION_START,
                tag_id=current_tag_id,
                session_id=current_session_id
            )
            await websocket.send(assignment_message.model_dump_json())

            user_context = SessionContext()
            user_context.session_id = current_session_id
            user_context.tag_id = current_tag_id
            session_manager.create_session(user_context)

            context = session_manager.get_session(constant.CHAT_ENGINE_NAME)
            vad_module: BaseVAD = context.global_module_manager.get_module("vad")
            asr_module: BaseASR = context.global_module_manager.get_module("asr")
            consumer = AudioConsumer(
                session_context=user_context,
                vad_module=vad_module,
                asr_module=asr_module,
                asr_result_callback=self.handle_asr_result,
                silence_timeout=1.0,
                max_buffer_duration=5.0,
            )
            consumer.start()
            self.audio_consumers[user_context.session_id] = consumer
            self.session_interrupt_flags[user_context.session_id] = False

            return current_tag_id, current_session_id
        return None, None

    async def _handle_client(self, websocket, path: Optional[str] = ""):
        current_tag_id, current_session_id = await self.register(websocket)
        if not current_session_id: return
        metadata = {"session_id": current_session_id, "tad_id": current_tag_id}
        try:
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    self.handle_audio(raw_message, metadata)
                else:
                    try:
                        if raw_message.strip().startswith('{'):
                            message_data = StreamEvent.model_validate_json(raw_message)
                            await self.handle_message(message_data, metadata)
                    except Exception as e:
                        logger.error(f"Error processing message: {e} | Raw: {raw_message}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            disconnected_session_id = self.websocket_to_session_id.pop(websocket, None)
            if disconnected_session_id:
                # 清理所有相关资源
                if disconnected_session_id in self.audio_consumers:
                    self.audio_consumers[disconnected_session_id].stop()
                    del self.audio_consumers[disconnected_session_id]
                if disconnected_session_id in self.session_interrupt_flags:
                    del self.session_interrupt_flags[disconnected_session_id]
                if disconnected_session_id in self.session_to_websocket:
                    del self.session_to_websocket[disconnected_session_id]
                if disconnected_session_id in self.session_turn_context:  # 新增：清理上下文
                    del self.session_turn_context[disconnected_session_id]

    def handle_audio(self, raw_message: bytes, metadata):
        session_id = metadata["session_id"]
        # 当收到新的音频时，设置中断标志
        if self.session_interrupt_flags.get(session_id, False) is False:
            # 只有在AI可能正在说话时（标志为False），用户的语音才算打断
            # 如果用户连续说话，第二次就不算打断第一次了
            if session_id in self.session_turn_context:
                self.session_turn_context[session_id]['was_interrupted'] = True
                logger.info(f"Interruption detected for session {session_id}. Marking context.")

        self.session_interrupt_flags[session_id] = True

        if session_id and session_id in self.audio_consumers:
            consumer = self.audio_consumers[session_id]
            consumer.process_chunk(raw_message)

    async def handle_message(self, message_data: StreamEvent, metadata: Optional[Dict]):
        session_id = metadata["session_id"]

        if message_data.event_type == EventType.CLIENT_SPEECH_END:
            logger.info(f"Handler received CLIENT_SPEECH_END for session {session_id}")
            if session_id in self.audio_consumers:
                self.audio_consumers[session_id].signal_client_speech_end()
        elif message_data.event_type == EventType.STREAM_END:
            logger.info(f"Client signaled stream end for session {session_id}.")
            if session_id in self.audio_consumers:
                self.audio_consumers[session_id].signal_client_speech_end()
        elif message_data.event_type == EventType.CLIENT_TEXT_INPUT:
            # 对于文本输入，我们认为它不会打断，直接重置上下文
            if session_id in self.session_turn_context:
                self.session_turn_context[session_id]['last_user_text'] = message_data.event_data.text
                self.session_turn_context[session_id]['was_interrupted'] = False
            await self.trigger_llm_and_tts(message_data.event_data, metadata)

    async def handle_asr_result(self, asr_event: StreamEvent, metadata: Optional[Dict]):
        session_id = metadata["session_id"]
        text_data: TextData = asr_event.event_data

        if text_data.is_final and text_data.text:
            turn_context = self.session_turn_context.get(session_id)

            if turn_context and turn_context['was_interrupted']:
                logger.info(f"Handling interrupted speech for session {session_id}.")
                # 拼接上一轮和本轮的文本
                combined_text = f"{turn_context.get('last_user_text', '')} {text_data.text}".strip()
                logger.info(f"Combined text for LLM: '{combined_text}'")
                llm_input_text = combined_text
            else:
                # 正常情况，直接使用本轮文本
                llm_input_text = text_data.text

            # 更新上下文，为下一轮做准备
            if turn_context:
                turn_context['last_user_text'] = llm_input_text  # 保存拼接后或独立的文本
                turn_context['was_interrupted'] = False  # 重置打断标志

            # 准备发送给LLM的数据
            llm_input_data = TextData(text=llm_input_text, is_final=True)

            # 重置TTS中断标志，准备播放新的回答
            self.session_interrupt_flags[session_id] = False

            await self.trigger_llm_and_tts(llm_input_data, metadata)

        elif text_data.is_final:
            logger.info(f"Final ASR result for session {session_id} is empty. Resetting interruption flag.")
            if session_id in self.session_turn_context:
                self.session_turn_context[session_id]['was_interrupted'] = False

    async def trigger_llm_and_tts(self, text_data: TextData, metadata: Optional[Dict]):
        session_id = metadata["session_id"]
        websocket = self.session_to_websocket.get(session_id)
        if not websocket: return

        context = session_manager.get_session(constant.CHAT_ENGINE_NAME)
        llm_module: BaseLLM = context.global_module_manager.get_module("llm")
        tts_module: BaseTTS = context.global_module_manager.get_module("tts")
        if not llm_module: return

        buffer = ""
        delimiters_pattern = re.compile(r'([，。！？；、,.!?;])')

        async for content in llm_module.stream_chat_response(text_data):
            if self.session_interrupt_flags.get(session_id, False):
                logger.info(f"TTS interrupted for session {session_id} by user speech.")
                break

            if content:
                buffer += content
                match = delimiters_pattern.search(buffer)
                if match:
                    sentence = buffer[:match.end()]
                    buffer = buffer[match.end():]
                    asyncio.create_task(self.process_and_send_tts_sentence(sentence, session_id, tts_module))

        if buffer and not self.session_interrupt_flags.get(session_id, False):
            asyncio.create_task(self.process_and_send_tts_sentence(buffer, session_id, tts_module, is_final=True))

    async def process_and_send_tts_sentence(self, sentence: str, session_id: str, tts_module: BaseTTS,
                                            is_final: bool = False):
        # ... 此函数逻辑保持不变 ...
        websocket = self.session_to_websocket.get(session_id)
        if not websocket or self.session_interrupt_flags.get(session_id, False): return

        await self._send_to_client(websocket, StreamEvent(event_type=EventType.SERVER_TEXT_RESPONSE,
                                                          event_data=TextData(text=sentence, is_final=is_final),
                                                          session_id=session_id).model_dump_json())

        audio_data = await tts_module.text_to_speech_block(TextData(text=sentence))
        if audio_data and audio_data.data and not self.session_interrupt_flags.get(session_id, False):
            audio_payload = StreamEvent(event_type=EventType.SERVER_AUDIO_RESPONSE, event_data=audio_data,
                                        session_id=session_id)
            await self._send_to_client(websocket, self.serialize_stream_event_with_audio(audio_payload))

    @staticmethod
    def serialize_stream_event_with_audio(event: StreamEvent):
        # ... 此函数逻辑保持不变 ...
        event_dict = event.model_dump()
        raw_data_bytes = event_dict['event_data']['data']
        base64_encoded_data = base64.b64encode(raw_data_bytes).decode('utf-8')
        event_dict['event_data']['data'] = base64_encoded_data
        return json.dumps(event_dict)

    @staticmethod
    async def _send_to_client(client_websocket, message: str):
        # ... 此函数逻辑保持不变 ...
        try:
            await client_websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")

    async def initialize(self):
        # ... 此函数逻辑保持不变 ...
        self.server = await websockets.serve(self._handle_client, self.host, self.port)
        print(f"[服务器] WebSocket 服务器已在 ws://{self.host}:{self.port} 启动。")
        await self.server.wait_closed()

    async def close(self):
        # ... 此函数逻辑保持不变 ...
        if self.server:
            self.server.close()
            await self.server.wait_closed()
