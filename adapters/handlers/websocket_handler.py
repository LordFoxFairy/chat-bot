import asyncio
import base64
import json
import re
import uuid
from typing import Optional, List, Dict

import websockets

import constant
from core.session_manager import session_manager
from data_models import StreamEvent, EventType, TextData
from modules import BaseLLM, BaseTTS
from modules.base_handler import BaseHandler
from utils.logging_setup import logger


class WebSocketServerHandler(BaseHandler):
    """
    一个用于处理 WebSocket 服务器的封装类。
    它负责启动服务器，处理客户端连接和消息，并支持固定的 tag_id 和动态的 session_id 跟踪。
    """

    def __init__(self, module_id, config):
        """
        初始化 WebSocket 服务器处理器。
        """
        super().__init__(module_id, config)
        self.host = self.handler_config.get("host")
        self.port = self.handler_config.get("port")
        self.server = None  # WebSocket 服务器实例

        # 核心映射：
        # tag_id 到其当前活跃的 session_id 的映射
        self.tag_to_current_session = {}  # {tag_id: current_session_id}
        # session_id 到具体的 WebSocket 连接对象的映射
        self.session_to_websocket = {}  # {session_id: websocket_object}
        # WebSocket 连接对象到其 session_id 的反向映射 (用于断开连接时快速查找 session_id)
        self.websocket_to_session_id = {}  # {websocket_object: session_id}

        # 在纯 asyncio 模式下，不需要手动管理事件循环和线程
        # self.loop = None
        # self.thread = None

        print(f"[服务器] WebSocketServerHandler 已初始化，监听地址: {self.host}:{self.port}")

    async def register(self, websocket):
        # 1. 接收客户端的注册消息，获取其 tag_id 和 last_session_id
        register_data = await websocket.recv()
        stream_event = StreamEvent.model_validate_json(register_data)
        if stream_event.event_type == EventType.SYSTEM_CLIENT_SESSION_START:
            client_tag_id = stream_event.tag_id
            client_last_session_id = stream_event.session_id

            if not client_tag_id:
                print(f"[服务器] 客户端 {websocket.remote_address} 未提供 tag_id，拒绝连接。")
                return  # 强制客户端提供 tag_id

            current_tag_id = client_tag_id
            # 为每个新的连接生成一个全新的 session_id
            current_session_id = str(uuid.uuid4())

            print(f"[服务器] 客户端 {websocket.remote_address} (Tag ID: {current_tag_id}) 尝试连接。")
            print(f"[服务器] 客户端 {current_tag_id} 上次已知 Session ID: {client_last_session_id}")

            # 2. 更新服务器的连接映射
            # 移除旧的连接（如果该 tag_id 之前有活跃的 session）
            if current_tag_id in self.tag_to_current_session:
                old_session_id = self.tag_to_current_session[current_tag_id]
                old_websocket = self.session_to_websocket.get(old_session_id)
                if old_websocket and old_websocket != websocket:
                    print(
                        f"[服务器] Tag ID {current_tag_id} 正在重新连接。关闭旧 Session {old_session_id} (旧连接 {old_websocket.remote_address})。")
                    try:
                        await old_websocket.close()  # 关闭旧连接，强制客户端断开
                    except Exception as e:
                        print(f"[服务器] 关闭旧连接 {old_websocket.remote_address} 时发生错误: {e}")
                    finally:
                        # 清理旧的映射
                        if old_session_id in self.session_to_websocket:
                            del self.session_to_websocket[old_session_id]
                        if old_websocket in self.websocket_to_session_id:
                            del self.websocket_to_session_id[old_websocket]

            # 更新为新的活跃 session
            self.tag_to_current_session[current_tag_id] = current_session_id
            self.session_to_websocket[current_session_id] = websocket
            self.websocket_to_session_id[websocket] = current_session_id

            print(f"[服务器] Tag ID {current_tag_id} 已分配新的 Session ID: {current_session_id}")
            print(f"[服务器] 当前活跃 Tag ID 数量: {len(self.tag_to_current_session)}")
            print(f"[服务器] 当前活跃 Session 数量: {len(self.session_to_websocket)}")

            # 3. 向客户端发送分配的 session_id
            assignment_message = StreamEvent(
                event_type=EventType.SYSTEM_SERVER_SESSION_START,
                tag_id=current_tag_id,
                session_id=current_session_id
            )
            await websocket.send(assignment_message.model_dump_json())
            print(
                f"[服务器] 已向 {websocket.remote_address} (Tag ID: {current_tag_id}) 发送 Session ID: {current_session_id}")

            return current_tag_id, current_session_id
        else:
            print(f"[服务器] 客户端 {websocket.remote_address} 发送了非注册消息作为第一条消息，关闭连接。")
            return  # 终止处理此客户端

    async def _handle_client(self, websocket, path: Optional[str] = ""):
        """
        处理单个客户端连接的异步协程。
        """

        current_tag_id = None
        current_session_id = None
        try:
            current_tag_id, current_session_id = await self.register(websocket)

            # 4. 持续处理来自客户端的后续消息
            async for raw_message in websocket:
                try:
                    message_data = StreamEvent.model_validate_json(raw_message)
                    sender_tag_id = message_data.tag_id
                    sender_session_id = message_data.session_id

                    # 验证消息是否来自当前活跃的 session
                    if sender_tag_id == current_tag_id and sender_session_id == current_session_id:
                        logger.info(
                            f"[服务器] 收到来自 Tag ID {sender_tag_id} (Session {sender_session_id}) 的消息: {message_data}")
                        metadata = {"session_id": sender_session_id, "tad_id": sender_tag_id}
                        await self.handle_message(message_data, metadata)

                    else:
                        print(
                            f"[服务器] 收到来自 Tag ID {sender_tag_id} (Session {sender_session_id}) 的无效或过期 Session 消息，内容: {raw_message}")

                except json.JSONDecodeError:
                    print(f"[服务器] 收到来自 {websocket.remote_address} 的非 JSON 格式消息: {raw_message}")
                except Exception as e:
                    print(f"[服务器] 处理来自 {websocket.remote_address} 的消息时发生错误: {e}")

        except websockets.exceptions.ConnectionClosedOK:
            print(f"[服务器] 客户端 {websocket.remote_address} 正常断开连接。")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"[服务器] 客户端 {websocket.remote_address} 意外断开连接: {e}")
        except Exception as e:
            print(f"[服务器] 处理客户端 {websocket.remote_address} 时发生错误: {e}")
        finally:
            # 客户端断开连接时，清理所有相关的映射
            if websocket in self.websocket_to_session_id:
                disconnected_session_id = self.websocket_to_session_id[websocket]
                # 只有当断开的这个 websocket 确实是该 session_id 对应的连接时才从 session_to_websocket 移除
                if disconnected_session_id in self.session_to_websocket and \
                        self.session_to_websocket[disconnected_session_id] == websocket:
                    del self.session_to_websocket[disconnected_session_id]
                    print(f"[服务器] Session {disconnected_session_id} 已从活跃 Session 中移除。")

                # 如果这个 session_id 恰好是它所属 tag_id 的 current_session，那么也需要更新 tag_to_current_session
                # 但这里我们采用的策略是：tag_to_current_session 总是指向最新建立的连接。
                # 如果这个 session_id 断开，而 tag_id 对应的 current_session_id 仍然是这个，
                # 那么意味着这个 tag_id 不再有活跃连接了。
                if current_tag_id and self.tag_to_current_session.get(current_tag_id) == disconnected_session_id:
                    del self.tag_to_current_session[current_tag_id]  #
                    # 移除这个 tag_id 的当前 session 映射
                    print(f"[服务器] Tag ID {current_tag_id} 的当前 Session 已断开。")

                del self.websocket_to_session_id[websocket]  # 移除 websocket 到 session_id 的映射

            print(f"[服务器] 客户端 {websocket.remote_address} 已断开连接。")
            print(f"[服务器] 当前活跃 Tag ID 数量: {len(self.tag_to_current_session)}")
            print(f"[服务器] 当前活跃 Session 数量: {len(self.session_to_websocket)}")

    async def handle_message(self, message_data: StreamEvent, metadata: Optional[Dict]):
        message_type = message_data.event_type
        if message_type == EventType.CLIENT_TEXT_INPUT:
            await self.handle_text(message_data, metadata)

    def serialize_stream_event_with_audio(self, event: StreamEvent):
        """
        将 StreamEvent 实例序列化为 JSON 字符串。
        如果 event_data 是 AudioData 且包含 bytes 数据，则将其 Base64 编码。

        Args:
            event: 待序列化的 StreamEvent 实例。

        Returns:
            序列化后的 JSON 字符串。
        """
        # 1. 将 StreamEvent 实例转换为 Python 字典
        # 使用 model_dump() 而不是 model_dump_json()，因为我们需要在转换为 JSON 前修改字典
        event_dict = event.model_dump()

        # 获取原始 bytes 数据
        raw_data_bytes = event_dict['event_data']['data']

        # 将 bytes 编码为 Base64 字符串
        base64_encoded_data = base64.b64encode(raw_data_bytes).decode('utf-8')

        # 更新字典中的 'data' 字段为 Base64 字符串
        event_dict['event_data']['data'] = base64_encoded_data

        return json.dumps(event_dict)

    async def handle_text(self, message_data: StreamEvent, metadata: Optional[Dict]) -> object:

        text_data: TextData = message_data.event_data
        context = session_manager.get_session(constant.CHAT_ENGINE_NAME)
        llm_module: BaseLLM = context.global_module_manager.get_module("llm")
        tts_module: BaseTTS = context.global_module_manager.get_module("tts")

        if not llm_module:
            return

        sentence_list: List[str] = []
        buffer = ""
        delimiters_pattern = re.compile(r'([，。！？；、,.!?;])')

        payload = StreamEvent(event_type=EventType.SERVER_TEXT_RESPONSE)
        audio = StreamEvent(event_type=EventType.SERVER_AUDIO_RESPONSE)
        audio.session_id = payload.session_id = metadata["session_id"]

        message_id = str(uuid.uuid4())

        # 2. 異步循環，實時解析句子
        async for content in llm_module.stream_chat_response(text_data):
            if content:
                buffer += content
                while True:
                    match = delimiters_pattern.search(buffer)

                    if match:
                        # 找到了一個完整的句子（包含標點）
                        complete_sentence = buffer[:match.end()]
                        sentence_list.append(complete_sentence)
                        buffer = buffer[match.end():]

                        payload.event_type = EventType.SERVER_TEXT_RESPONSE
                        payload.event_data = TextData(text=complete_sentence, is_final=False, message_id=message_id)
                        await self._broadcast_message(payload.model_dump_json())


                        audio_data = await tts_module.text_to_speech_block(payload.event_data)
                        audio_data.message_id = message_id
                        audio.event_data = audio_data
                        await self._broadcast_message(self.serialize_stream_event_with_audio(audio))

                    else:
                        break

        if buffer:
            payload.event_data = TextData(text=buffer, is_final=True)
            await self._broadcast_message(payload.model_dump_json())

            sentence_list.append(buffer)
            buffer = ""  # 清空緩衝區

        else:
            payload.event_data = TextData(text=buffer, is_final=True)
            await self._broadcast_message(payload.model_dump_json())


        audio_data = await tts_module.text_to_speech_block(payload.event_data)
        audio_data.message_id = message_id
        audio_data.is_final = True
        audio.event_data = audio_data
        await self._broadcast_message(self.serialize_stream_event_with_audio(payload))

        logger.info(f"流式解析完成，共分割成 {len(sentence_list)} 個段落。")
        logger.info(f"文本内容生成： {' '.join(sentence_list)}")

    async def _broadcast_message(self, message):
        """
        向所有活跃连接的客户端广播消息。
        """

        logger.info(message)
        if self.session_to_websocket:
            # 获取所有当前活跃的 WebSocket 连接对象
            websockets_to_send = list(self.session_to_websocket.values())
            await asyncio.gather(*[
                self._send_to_client(client_ws, message) for client_ws in websockets_to_send
            ], return_exceptions=True)  # return_exceptions=True 允许部分失败而不中断所有发送

    @staticmethod
    async def _send_to_client(client_websocket, message: str):
        """
        发送消息给单个客户端。
        """
        try:
            await client_websocket.send(message)
        except websockets.exceptions.ConnectionClosedOK:
            # 客户端已正常关闭，无需再次尝试发送
            pass
        except websockets.exceptions.ConnectionClosedError:
            # 客户端异常关闭，可能已经在 _handle_client 的 finally 块中处理了移除
            print(f"[服务器] 客户端 {client_websocket.remote_address} 异常关闭，尝试发送失败。")
        except Exception as e:
            print(f"[服务器] 向客户端 {client_websocket.remote_address} 发送消息失败: {e}")

    async def initialize(self):
        """
        启动 WebSocket 服务器。这是一个异步协程，它将阻塞直到服务器关闭。
        """
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        print(f"[服务器] WebSocket 服务器已在 ws://{self.host}:{self.port} 启动。")
        await self.server.wait_closed()  # 等待服务器关闭

    async def close(self):
        """
        异步关闭 WebSocket 服务器。
        """
        if self.server:
            print("[服务器] 正在关闭 WebSocket 服务器...")
            self.server.close()  # 关闭服务器
            await self.server.wait_closed()  # 等待服务器完全关闭
            print("[服务器] WebSocket 服务器已关闭。")
        else:
            print("[服务器] 没有活动的 WebSocket 服务器可关闭。")
