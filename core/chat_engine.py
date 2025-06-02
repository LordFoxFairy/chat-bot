# core_framework/chat_engine.py
import asyncio
import logging
import json
import os
import time
import uuid  # 用于在客户端未提供有效UUID时生成备用ID
from typing import Dict, Any, Optional, List  # 确保 List 已导入

import websockets

from services.config_loader import ConfigLoader
from utils.context_utils import current_user_session_var
from .module_manager import ModuleManager
from .session_context import SessionContext  # 新的会话上下文类

# 模块基类，用于类型检查和获取模块实例
from modules.base_asr import BaseASR
from modules.base_llm import BaseLLM
from modules.base_tts import BaseTTS
from modules.base_vad import BaseVAD
from data_models.text_data import TextData
from data_models.audio_data import AudioData

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TIMEOUT_SECONDS_CE = 30 * 60  # 默认会话超时时间


class ChatEngine:
    """
    全局聊天引擎 (ChatEngine)。
    负责初始化所有模块、管理所有用户会话 (SessionContext)、
    并编排每个会话内的对话流程。
    """

    def __init__(self,
                 config: Dict[str, Any],  # 直接接收加载好的全局配置字典
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化全局 ChatEngine。

        Args:
            config (Dict[str, Any]): 已加载的全局应用配置。
            loop (Optional[asyncio.AbstractEventLoop]): 事件循环。
        """
        self.global_config = config
        self.loop = loop if loop else asyncio.get_event_loop()

        self.module_manager = ModuleManager(
            config=self.global_config,
            loop=self.loop
        )

        self.active_sessions: Dict[str, SessionContext] = {}

        system_settings = self.global_config.get("system_config", {})
        self.session_timeout_seconds = system_settings.get("session_timeout_seconds",
                                                           DEFAULT_SESSION_TIMEOUT_SECONDS_CE)

        self._cleanup_task: Optional[asyncio.Task] = None
        self.is_shutting_down = False

        logger.info("全局 ChatEngine 初始化完成。")

    async def initialize(self):
        """
        异步初始化 ChatEngine，主要是初始化 ModuleManager。
        """
        logger.info("ChatEngine 正在进行异步初始化 (初始化模块)...")
        await self.module_manager.initialize_modules()
        logger.info("ChatEngine 的 ModuleManager 初始化完成。")

        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = self.loop.create_task(self._periodic_session_cleanup())
            logger.info("ChatEngine 的会话超时清理任务已启动。")

    def _get_module_instance(self, module_type: str, session_context: SessionContext) -> Optional[Any]:
        """
        辅助函数，根据会话配置获取特定类型的模块实例。
        """
        return self.module_manager.get_module(module_type)

    async def _handle_text_input(self, session_context: SessionContext, text_input: str):
        """处理来自特定会话的文本输入。"""
        logger.info(f"[{session_context.session_id}] ChatEngine 正在处理文本: '{text_input[:50]}...'")

        llm_module:BaseLLM = self._get_module_instance("llm", session_context)
        llm_response_text = f"LLM模块未配置。您说的是: '{text_input}'"

        if llm_module:
            try:
                llm_input_text_data = TextData(text=text_input, session_id=session_context.session_id)
                response_data = await llm_module.generate_complete_response(llm_input_text_data)
                if isinstance(response_data, TextData):
                    llm_response_text = response_data.text
                elif isinstance(response_data, str):
                    llm_response_text = response_data
                else:
                    logger.error(f"[{session_context.session_id}] LLM模块返回了非预期的类型: {type(response_data)}")
            except Exception as e_llm:
                logger.error(f"[{session_context.session_id}] LLM处理时出错: {e_llm}", exc_info=True)
                llm_response_text = f"抱歉，我在理解您的意思时遇到了问题。您说的是: '{text_input}'"

        await session_context.send_message_to_client({
            "type": "text_response",
            "content": llm_response_text,
            "session_id": session_context.session_id
        })

        tts_module: BaseTTS = self._get_module_instance("tts", session_context)
        if tts_module:
            try:
                tts_input_text_data = TextData(text=llm_response_text, session_id=session_context.session_id)
                async for audio_segment in tts_module.text_to_speech_stream(tts_input_text_data):
                    if audio_segment and audio_segment.data:
                        await session_context.send_audio_to_client(audio_segment.data)
                await session_context.send_message_to_client(
                    {"type": "audio_stream_end", "message": "TTS stream complete."})
            except Exception as e_tts:
                logger.error(f"[{session_context.session_id}] TTS合成时出错: {e_tts}", exc_info=True)
                await session_context.send_message_to_client({"type": "error", "message": "TTS合成失败。"})
        else:
            logger.warning(f"[{session_context.session_id}] 未找到TTS模块，无法合成语音。")
            await session_context.send_message_to_client({"type": "info", "message": "TTS服务不可用。"})

    async def _handle_audio_chunk(self, session_context: SessionContext, audio_chunk: bytes):
        """处理来自特定会话的音频块。"""
        logger.debug(f"[{session_context.session_id}] ChatEngine 收到音频块，长度: {len(audio_chunk)}")
        await session_context.send_message_to_client({
            "type": "info",
            "content": f"收到 {len(audio_chunk)} 字节的音频数据块。"
        })

    async def _handle_audio_end(self, session_context: SessionContext):
        """处理来自特定会话的音频流结束信号。"""
        logger.info(f"[{session_context.session_id}] ChatEngine 收到音频流结束信号。")
        await session_context.send_message_to_client({
            "type": "info",
            "content": "音频流已结束，正在处理您的完整语音..."
        })
        await asyncio.sleep(1)
        final_response_text = "这是对您完整语音的模拟回复。"
        await self._handle_text_input(session_context, final_response_text)

    async def _process_message_for_session(self, session_context: SessionContext, payload: Any, origin_type: str):
        """
        在特定会话的上下文中处理消息。
        """
        session_ctx_token = current_user_session_var.set(session_context)
        logger.debug(f"[{session_context.session_id}] ContextVar 'current_user_session' 已设置为当前会话。")

        try:
            # --- 修正 origin_type 的处理逻辑 ---
            if origin_type == "text":  # 标准化的文本输入类型
                if isinstance(payload, str):
                    await self._handle_text_input(session_context, payload)
                else:
                    logger.warning(
                        f"[{session_context.session_id}] 收到类型为 'text' 的请求，但载荷不是字符串: {type(payload)}")
                    await session_context.send_message_to_client({"type": "error", "message": "无效的文本输入格式。"})

            elif origin_type == "audio_chunk":  # 标准化的音频块类型
                if isinstance(payload, bytes):
                    await self._handle_audio_chunk(session_context, payload)
                else:
                    logger.warning(
                        f"[{session_context.session_id}] 收到类型为 'audio_chunk' 的请求，但载荷不是字节: {type(payload)}")
                    await session_context.send_message_to_client({"type": "error", "message": "无效的音频数据格式。"})

            elif origin_type == "audio_end":  # 标准化的音频结束信号类型
                await self._handle_audio_end(session_context)

            # raw_text 可以被视为一种特殊的 "text" 输入
            elif origin_type == "raw_text":
                logger.info(f"[{session_context.session_id}] ChatEngine 正在处理原始文本输入: {str(payload)[:50]}...")
                await self._handle_text_input(session_context, str(payload))

            else:  # 处理其他未明确映射的 origin_type
                logger.warning(
                    f"[{session_context.session_id}] ChatEngine 收到未明确处理的 origin_type: '{origin_type}'。原始载荷: {str(payload)[:100]}")
                await session_context.send_message_to_client(
                    {"type": "error", "message": f"服务器无法处理类型为 '{origin_type}' 的请求。"})

        except Exception as e:
            logger.error(f"[{session_context.session_id}] 处理消息时发生严重错误: {e}", exc_info=True)
            try:
                await session_context.send_message_to_client({"type": "error", "message": "服务器内部处理错误。"})
            except Exception as e_send:
                logger.error(f"[{session_context.session_id}] 发送错误消息给客户端失败: {e_send}")
        finally:
            current_user_session_var.reset(session_ctx_token)
            logger.debug(f"[{session_context.session_id}] ContextVar 'current_user_session' 已重置。")

    async def handle_websocket_connection(self, websocket: Any, path: Optional[str] = None):
        """
        处理新的 WebSocket 连接。
        为每个连接创建或获取一个 SessionContext，并在该上下文中处理消息。
        """
        client_address = websocket.remote_address
        session_context: Optional[SessionContext] = None
        assigned_session_id: Optional[str] = None

        logger.info(f"客户端 {client_address} 尝试通过路径 '{path}' 连接...")

        try:
            external_session_id_from_client = None
            user_specific_config_data = {}
            first_message_payload = None  # 用于存储非init_session的第一条消息
            first_message_type_client = None  # 用于存储非init_session的第一条消息的类型

            try:
                initial_message_str = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                if isinstance(initial_message_str, str):
                    initial_data = json.loads(initial_message_str)
                    if initial_data.get("type") == "init_session":
                        external_session_id_from_client = initial_data.get("session_id")
                        user_specific_config_data = initial_data.get("config", {})
                        logger.info(
                            f"客户端 {client_address} 发送初始化请求，session_id: {external_session_id_from_client}")
                    else:
                        logger.warning(
                            f"客户端 {client_address} 的第一条消息不是 'init_session'。将为其生成新会话ID。消息: {initial_message_str[:100]}")
                        first_message_payload = initial_data.get("text") or initial_data.get("content")
                        first_message_type_client = initial_data.get("type")  # 保存客户端原始类型
                else:
                    logger.warning(f"客户端 {client_address} 的第一条消息不是文本。将为其生成新会话ID。")
                    # 如果第一条是二进制，则认为不是初始化消息
            except asyncio.TimeoutError:
                logger.warning(f"等待客户端 {client_address} 初始化消息超时。将为其生成新会话ID。")
            except json.JSONDecodeError:
                logger.warning(f"客户端 {client_address} 的初始化消息不是有效的JSON。将为其生成新会话ID。")
            except Exception as e_init:
                logger.error(f"处理客户端 {client_address} 初始化消息时出错: {e_init}", exc_info=True)
                await websocket.close(code=1002, reason="无效的初始化请求")
                return

            if external_session_id_from_client:
                try:  # 校验客户端提供的UUID格式
                    uuid.UUID(external_session_id_from_client)
                except ValueError:
                    logger.warning(
                        f"客户端提供的 session_id '{external_session_id_from_client}' 不是有效的UUID格式。将生成新的UUID。")
                    external_session_id_from_client = str(uuid.uuid4())

                session_context = self.active_sessions.get(external_session_id_from_client)
                if session_context:
                    logger.info(f"客户端 {client_address} 重新连接到现有会话: {external_session_id_from_client}")
                    session_context.websocket = websocket
                    session_context.update_activity_time()
                    assigned_session_id = external_session_id_from_client
                else:
                    logger.info(
                        f"为客户端 {client_address} 使用其提供的ID创建新会话: {external_session_id_from_client}")
                    session_context = SessionContext(
                        session_id=external_session_id_from_client,
                        websocket=websocket,
                        global_config=self.global_config,
                        user_specific_config=user_specific_config_data
                    )
                    self.active_sessions[external_session_id_from_client] = session_context
                    assigned_session_id = external_session_id_from_client
            else:
                new_session_id = str(uuid.uuid4())
                logger.info(f"为客户端 {client_address} 生成新的会话ID: {new_session_id}")
                session_context = SessionContext(
                    session_id=new_session_id,
                    websocket=websocket,
                    global_config=self.global_config,
                    user_specific_config=user_specific_config_data  # 即使没有session_id，也可能从init消息中获取config
                )
                self.active_sessions[new_session_id] = session_context
                assigned_session_id = new_session_id

            if not session_context or not assigned_session_id:
                logger.error(f"为客户端 {client_address} 创建/获取会话上下文失败。")
                await websocket.close(code=1011, reason="服务器会话处理错误")
                return

            await websocket.send(json.dumps({
                "type": "session_established",
                "session_id": assigned_session_id,
                "message": "会话已成功建立。"
            }))

            # 如果第一条消息不是 init_session，并且已保存，现在处理它
            if first_message_payload is not None and first_message_type_client:
                logger.info(
                    f"[{assigned_session_id}] 处理之前收到的非init消息: type='{first_message_type_client}', payload='{str(first_message_payload)[:50]}...'")
                # --- 映射客户端类型到内部 origin_type ---
                internal_origin_type = "unknown_client_type"  # 默认
                if first_message_type_client in ["text_message", "tts_request", "chat_message",
                                                 "text_input_from_client"]:
                    internal_origin_type = "text"
                elif first_message_type_client == "audio_stream_end_signal":
                    internal_origin_type = "audio_end"
                # ... 其他映射 ...
                await self._process_message_for_session(session_context, first_message_payload, internal_origin_type)

            async for message in websocket:
                session_context.update_activity_time()

                # --- 映射客户端类型到内部 origin_type ---
                internal_origin_type = "unknown_client_type"  # 默认
                payload_to_process = None

                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        payload_to_process = data.get("text") or data.get("content")
                        client_message_type = data.get("type")

                        if client_message_type in ["text_message", "tts_request", "chat_message",
                                                   "text_input_from_client"]:
                            internal_origin_type = "text"
                        elif client_message_type == "audio_stream_end_signal":
                            internal_origin_type = "audio_end"
                            payload_to_process = None  # audio_end 通常没有payload
                        # ... 其他 client_message_type 到 internal_origin_type 的映射 ...

                        if payload_to_process is not None or internal_origin_type == "audio_end":
                            await self._process_message_for_session(session_context, payload_to_process,
                                                                    internal_origin_type)
                        elif client_message_type and not payload_to_process and internal_origin_type == "text":
                            logger.warning(
                                f"[{assigned_session_id}] 文本类型消息 '{client_message_type}' 缺少有效载荷。")
                            await session_context.send_message_to_client({"type": "error",
                                                                          "message": f"消息类型 '{client_message_type}' 需要 'text' 或 'content' 字段。"})
                        else:  # 如果 internal_origin_type 仍然是 "unknown_client_type"
                            logger.warning(
                                f"[{assigned_session_id}] 收到未映射的客户端JSON消息类型: {client_message_type}")
                            await self._process_message_for_session(session_context, data, "unknown_client_type")


                    except json.JSONDecodeError:
                        logger.warning(f"[{assigned_session_id}] 收到的文本消息非JSON格式: {message[:100]}。")
                        await self._process_message_for_session(session_context, message, "raw_text")
                elif isinstance(message, bytes):
                    await self._process_message_for_session(session_context, message, "audio_chunk")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端 {client_address} (会话: {assigned_session_id or 'N/A'}) 连接已关闭。")
        except Exception as e_conn:
            logger.error(f"处理客户端 {client_address} (会话: {assigned_session_id or 'N/A'}) 连接时发生错误: {e_conn}",
                         exc_info=True)
            if websocket.open:  # 检查 websocket 是否仍然是 open 状态
                await websocket.close(code=1011, reason="服务器内部错误")
        finally:
            if assigned_session_id and assigned_session_id in self.active_sessions:
                logger.info(f"正在清理会话 '{assigned_session_id}'...")
                session_to_close = self.active_sessions.pop(assigned_session_id, None)
                if session_to_close:
                    await session_to_close.close(reason="WebSocket 连接结束")
            else:
                logger.info(
                    f"客户端 {client_address} 断开连接，但未找到关联的活动会话 '{assigned_session_id}' 进行清理。")

    async def _periodic_session_cleanup(self):
        """定期清理超时的会话。"""
        logger.info("ChatEngine 的会话超时定期清理任务已启动。")
        while not self.is_shutting_down:
            await asyncio.sleep(max(10, self.session_timeout_seconds // 5))
            if self.is_shutting_down: break

            now = time.time()
            expired_session_ids: List[str] = []
            current_sessions_snapshot = list(self.active_sessions.items())
            for sid, session_ctx in current_sessions_snapshot:
                if (now - session_ctx.last_activity_time) > self.session_timeout_seconds:
                    expired_session_ids.append(sid)

            if expired_session_ids:
                logger.info(f"发现 {len(expired_session_ids)} 个超时会话: {expired_session_ids}。正在清理...")
                for sid_to_delete in expired_session_ids:
                    if sid_to_delete in self.active_sessions:
                        session_to_close = self.active_sessions.pop(sid_to_delete, None)
                        if session_to_close:
                            logger.info(f"正在关闭超时的会话 '{sid_to_delete}'...")
                            await session_to_close.close(reason="超时")
        logger.info("ChatEngine 的会话超时清理任务已停止。")

    async def shutdown(self):
        """关闭 ChatEngine，清理所有资源。"""
        logger.info("全局 ChatEngine 正在关闭...")
        self.is_shutting_down = True

        if self._cleanup_task and not self._cleanup_task.done():
            logger.info("正在取消会话清理任务...")
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.info("会话清理任务已取消。")
            self._cleanup_task = None

        active_session_ids_to_close = list(self.active_sessions.keys())
        if active_session_ids_to_close:
            logger.info(f"正在关闭 {len(active_session_ids_to_close)} 个剩余活动会话...")
            closing_tasks = []
            for session_id in active_session_ids_to_close:
                session_ctx = self.active_sessions.pop(session_id, None)
                if session_ctx:
                    closing_tasks.append(session_ctx.close("ChatEngine 关闭"))
            if closing_tasks:
                await asyncio.gather(*closing_tasks, return_exceptions=True)

        self.active_sessions.clear()

        if self.module_manager:
            logger.info("正在关闭 ModuleManager...")
            await self.module_manager.shutdown_modules()

        logger.info("全局 ChatEngine 关闭完成。")

    @classmethod
    async def create_and_run(cls, config_file_path: str, host: str, port: int):
        """
        类方法，作为启动 ChatEngine 服务的入口点。
        """
        logger.info(f"ChatEngine 服务正在尝试从配置文件 '{config_file_path}' 启动...")
        if not os.path.exists(config_file_path):
            logger.critical(f"错误: 配置文件 '{config_file_path}' 未找到。服务无法启动。")
            return

        try:
            config_loader = ConfigLoader()
            global_app_config = config_loader.load_config(config_file_path)
            if not global_app_config:
                logger.critical(f"错误: 从 '{config_file_path}' 加载配置失败。服务无法启动。")
                return

            engine = cls(config=global_app_config, loop=asyncio.get_event_loop())
            await engine.initialize()
            logger.info("全局 ChatEngine 初始化成功。")

            server_instance = None
            try:
                logger.info(f"准备在 ws://{host}:{port} 上启动 WebSocket 服务器，并由 ChatEngine 处理连接...")
                server_instance = await websockets.serve(engine.handle_websocket_connection, host, port)
                logger.info(f"WebSocket 服务器已在 ws://{host}:{port} 上启动，由 ChatEngine 管理。")
                await asyncio.Future()
            except OSError as e_os:
                logger.critical(f"启动 WebSocket 服务器失败 (例如端口被占用): {e_os}", exc_info=True)
            except Exception as e_serve:
                logger.critical(f"运行 WebSocket 服务器时发生未知错误: {e_serve}", exc_info=True)
            finally:
                if server_instance:
                    logger.info("正在关闭 WebSocket 服务器...")
                    server_instance.close()
                    await server_instance.wait_closed()
                logger.info("正在关闭 ChatEngine 服务...")
                await engine.shutdown()
                logger.info("ChatEngine 服务已关闭。")

        except Exception as e_outer:
            logger.critical(f"ChatEngine 服务启动或运行期间发生顶层错误: {e_outer}", exc_info=True)
