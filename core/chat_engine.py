# chat-bot/core/chat_engine.py
import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from typing import Dict, Any, Optional, List, AsyncGenerator

import websockets

from data_models.audio_data import AudioData, AudioFormat
from data_models.text_data import TextData
# 模块基类和数据模型导入
from modules.base_asr import BaseASR
from modules.base_llm import BaseLLM
from modules.base_tts import BaseTTS
from modules.base_vad import BaseVAD
# 内部导入
from services.config_loader import ConfigLoader
from utils.context_utils import current_user_session_var
from .module_manager import ModuleManager
from .session_context import SessionContext

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TIMEOUT_SECONDS_CE = 30 * 60
DEFAULT_ACTIVATION_TIMEOUT_CHECK_INTERVAL_SECONDS = 5
DEFAULT_MIN_SILENCE_MS_FOR_EOS = 1200  # 默认语句结束的静默阈值 (毫秒)
DEFAULT_MAX_SPEECH_SEGMENT_MS = 5000  # 默认最大语音段长度 (毫秒)
TTS_SENTENCE_BUFFER_TIMEOUT = 0.2
TTS_MIN_CHARS_FOR_SYNTHESIS = 5


class ChatEngine:
    def __init__(self,
                 config: Dict[str, Any],
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        self.global_config = config
        self.loop = loop if loop else asyncio.get_event_loop()
        self.module_manager = ModuleManager(config=self.global_config, loop=self.loop)
        self.active_sessions: Dict[str, SessionContext] = {}

        system_settings = self.global_config.get("system_config", {})
        self.session_timeout_seconds = system_settings.get("session_timeout_seconds",
                                                           DEFAULT_SESSION_TIMEOUT_SECONDS_CE)

        self.default_min_silence_ms_for_eos: int = DEFAULT_MIN_SILENCE_MS_FOR_EOS
        self.default_max_speech_segment_ms: int = DEFAULT_MAX_SPEECH_SEGMENT_MS

        vad_module_global_config = self.global_config.get("modules", {}).get("vad", {})
        if vad_module_global_config:  # pragma: no cover
            vad_adapter_name = vad_module_global_config.get("adapter_type")
            if vad_adapter_name:
                adapter_config = vad_module_global_config.get("config", {}).get(vad_adapter_name, {})
                self.default_min_silence_ms_for_eos = int(
                    adapter_config.get("min_silence_duration_ms_eos",
                                       adapter_config.get("min_silence_duration_ms",
                                                          self.default_min_silence_ms_for_eos))
                )
                self.default_max_speech_segment_ms = int(
                    adapter_config.get("max_speech_segment_duration_ms", self.default_max_speech_segment_ms)
                )
        logger.info(
            f"ChatEngine: 默认语句结束静默阈值: {self.default_min_silence_ms_for_eos} ms, 默认最大语音段: {self.default_max_speech_segment_ms} ms")

        self._activation_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self.is_shutting_down = False
        logger.info("全局 ChatEngine 初始化完成。")

    async def initialize(self):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        logger.info("ChatEngine 正在进行异步初始化 (初始化模块)...")
        await self.module_manager.initialize_modules()
        logger.info("ChatEngine 的 ModuleManager 初始化完成。")
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = self.loop.create_task(self._periodic_session_cleanup())
            logger.info("ChatEngine 的会话超时清理任务已启动。")
        if self._activation_check_task is None or self._activation_check_task.done():
            self._activation_check_task = self.loop.create_task(self._periodic_activation_timeout_check())
            logger.info("ChatEngine 的激活状态超时检查任务已启动。")

    def _get_module_instance(self, module_type: str) -> Optional[Any]:
        return self.module_manager.get_module(module_type)

    async def _send_text_chunk_to_client(self, session_context: SessionContext, text_chunk: str,
                                         is_final_llm_chunk: bool):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        await session_context.send_message_to_client({
            "type": "text_chunk", "content": text_chunk,
            "session_id": session_context.session_id, "is_final": is_final_llm_chunk
        })

    async def _send_audio_chunk_to_client(self, session_context: SessionContext, audio_data: AudioData):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        if audio_data and audio_data.data:
            await session_context.send_audio_to_client(audio_data.data)
        if audio_data.is_final:
            logger.debug(f"[{session_context.session_id}] TTS音频片段 (is_final={audio_data.is_final}) 已发送。")

    async def _send_system_or_error_reply(self, session_context: SessionContext, reply_text: str,
                                          message_type: str = "system_message"):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        await session_context.send_message_to_client({
            "type": message_type, "content": reply_text, "session_id": session_context.session_id
        })
        tts_module: Optional[BaseTTS] = self._get_module_instance("tts")
        if tts_module and tts_module.is_ready:
            try:
                tts_params = {};
                session_tts_adapter_name = tts_module.enabled_adapter_name
                session_tts_config = session_context.session_config.get("modules", {}).get("tts", {}).get("config",
                                                                                                          {}).get(
                    session_tts_adapter_name, {})
                if "voice" in session_tts_config: tts_params["voice"] = session_tts_config["voice"]
                tts_input_text_data = TextData(text=reply_text, chunk_id=session_context.session_id)
                async for audio_segment in tts_module.text_to_speech_stream(tts_input_text_data, **tts_params):
                    await self._send_audio_chunk_to_client(session_context, audio_segment)
                await session_context.send_message_to_client(
                    {"type": "audio_stream_end", "message": f"TTS for '{message_type}' complete."})
            except Exception as e_tts:
                logger.error(f"[{session_context.session_id}] TTS合成系统/错误消息时出错: {e_tts}", exc_info=True)

    async def _tts_consumer_task(self, session_context: SessionContext, tts_input_queue: asyncio.Queue[Optional[str]]):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        log_prefix = f"ChatEngine [{session_context.session_id}] _tts_consumer"
        tts_module: Optional[BaseTTS] = self._get_module_instance("tts")
        if not (tts_module and tts_module.is_ready):
            logger.warning(f"{log_prefix}: TTS模块不可用，此任务将不执行任何操作。")
            while True:
                try:
                    text_to_synthesize = await asyncio.wait_for(tts_input_queue.get(),
                                                                timeout=0.1); tts_input_queue.task_done();
                except (asyncio.TimeoutError, Exception):
                    break
                if text_to_synthesize is None: break
            return
        try:
            while True:
                try:
                    text_to_synthesize = await asyncio.wait_for(tts_input_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if session_context.llm_producer_task and session_context.llm_producer_task.done(): logger.info(
                        f"{log_prefix}: LLM生产者任务已结束，TTS消费者准备退出。"); break
                    continue
                if text_to_synthesize is None: logger.info(
                    f"{log_prefix}: 收到结束信号，TTS处理完成。"); tts_input_queue.task_done(); break
                if not text_to_synthesize.strip(): tts_input_queue.task_done(); continue
                logger.info(f"{log_prefix}: 准备合成语音: '{text_to_synthesize[:30]}...'")
                tts_params = {};
                session_tts_adapter_name = tts_module.enabled_adapter_name
                session_tts_config = session_context.session_config.get("modules", {}).get("tts", {}).get("config",
                                                                                                          {}).get(
                    session_tts_adapter_name, {})
                if "voice" in session_tts_config: tts_params["voice"] = session_tts_config["voice"]
                tts_input_text_data = TextData(text=text_to_synthesize,
                                               chunk_id=f"{session_context.session_id}_tts_{time.time_ns()}")
                async for audio_chunk in tts_module.text_to_speech_stream(tts_input_text_data, **tts_params):
                    await self._send_audio_chunk_to_client(session_context, audio_chunk)
                tts_input_queue.task_done();
                logger.debug(f"{log_prefix}: 完成句子 '{text_to_synthesize[:30]}...' 的TTS。")
        except asyncio.CancelledError:
            logger.info(f"{log_prefix}: TTS消费者任务被取消。")
        except Exception as e:
            logger.error(f"{log_prefix}: TTS消费者任务发生错误: {e}", exc_info=True)
        finally:
            logger.info(f"{log_prefix}: TTS消费者任务结束。")
            while not tts_input_queue.empty():
                try:
                    item = tts_input_queue.get_nowait(); tts_input_queue.task_done()
                except (asyncio.QueueEmpty, Exception):
                    break
                if item is not None: logger.warning(f"{log_prefix}: 清理未处理的TTS输入: {item[:30]}...")

    async def _llm_to_tts_producer_task(self, session_context: SessionContext,
                                        llm_output_stream: AsyncGenerator[TextData, None],
                                        tts_input_queue: asyncio.Queue[Optional[str]],
                                        text_chunk_queue_for_client: asyncio.Queue[Optional[str]]):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        log_prefix = f"ChatEngine [{session_context.session_id}] _llm_to_tts_producer"
        sentence_buffer = "";
        full_llm_response_text = ""
        try:
            async for llm_text_chunk_data in llm_output_stream:
                if not isinstance(llm_text_chunk_data, TextData) or not llm_text_chunk_data.text:
                    if llm_text_chunk_data and llm_text_chunk_data.is_final:
                        logger.debug(f"{log_prefix}: LLM流结束（空final块）。")
                        if not session_context._client_text_queue_ended: await text_chunk_queue_for_client.put(
                            None); session_context._client_text_queue_ended = True
                    continue
                chunk_text = llm_text_chunk_data.text;
                full_llm_response_text += chunk_text
                if not session_context._client_text_queue_ended: await text_chunk_queue_for_client.put(chunk_text)
                sentence_buffer += chunk_text
                parts = re.split(r'(?<=[。！？!?…])', sentence_buffer)
                if len(parts) > 1:
                    for i in range(len(parts) - 1):
                        sentence = parts[i].strip()
                        if sentence: logger.debug(
                            f"{log_prefix}: 提取到句子送TTS: '{sentence[:30]}...'"); await tts_input_queue.put(sentence)
                    sentence_buffer = parts[-1]
                elif len(sentence_buffer) >= TTS_MIN_CHARS_FOR_SYNTHESIS * 10:
                    logger.debug(f"{log_prefix}: 缓冲区文本较长且无标点，尝试发送: '{sentence_buffer[:30]}...'")
                    await tts_input_queue.put(sentence_buffer);
                    sentence_buffer = ""
                if llm_text_chunk_data.is_final:
                    logger.debug(f"{log_prefix}: 收到LLM的is_final=True文本块。")
                    if not session_context._client_text_queue_ended: await text_chunk_queue_for_client.put(
                        None); session_context._client_text_queue_ended = True
                    break
            if sentence_buffer.strip():
                logger.debug(f"{log_prefix}: LLM流结束，处理剩余缓冲区文本送TTS: '{sentence_buffer[:30]}...'")
                await tts_input_queue.put(sentence_buffer.strip())
        #     if full_llm_response_text.strip():
        #         logger.info(f"{log_prefix}: LLM完整响应文本 (将存入历史): '{full_llm_response_text[:100]}...'")
        #         llm_module: Optional[BaseLLM] = self._get_module_instance("llm")  # type: ignore
        #         if llm_module and hasattr(llm_module, 'session_manager') and llm_module.session_manager:  # type: ignore
        #             llm_module.session_manager.add_to_dialogue_history(session_context.session_id, {"role": "assistant",
        #                                                                                             "content": full_llm_response_text})  # type: ignore
        #         elif session_context.dialog_history is not None:
        #             session_context.dialog_history.append({"role": "assistant", "content": full_llm_response_text})
        except asyncio.CancelledError:
            logger.info(f"{log_prefix}: LLM到TTS生产者任务被取消。")
        except Exception as e:
            logger.error(f"{log_prefix}: LLM到TTS生产者任务发生错误: {e}", exc_info=True)
        finally:
            logger.info(f"{log_prefix}: LLM到TTS生产者任务结束。向消费者发送结束信号。")
            await tts_input_queue.put(None)
            if not session_context._client_text_queue_ended:
                try:
                    await text_chunk_queue_for_client.put(None); session_context._client_text_queue_ended = True
                except Exception as e_q_final:
                    logger.error(f"{log_prefix}: 发送最终None到客户端文本队列失败: {e_q_final}")

    async def _client_text_sender_task(self, session_context: SessionContext,
                                       text_chunk_queue: asyncio.Queue[Optional[str]]):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        log_prefix = f"ChatEngine [{session_context.session_id}] _client_text_sender"
        session_context._client_text_queue_ended = False
        try:
            while True:
                text_chunk = await text_chunk_queue.get()
                if text_chunk is None:
                    logger.info(f"{log_prefix}: 收到文本流结束信号。")
                    await self._send_text_chunk_to_client(session_context, "", True)
                    text_chunk_queue.task_done();
                    session_context._client_text_queue_ended = True;
                    break
                await self._send_text_chunk_to_client(session_context, text_chunk, False)
                text_chunk_queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"{log_prefix}: 客户端文本发送任务被取消。")
        except Exception as e:
            logger.error(f"{log_prefix}: 客户端文本发送任务发生错误: {e}", exc_info=True)
        finally:
            logger.info(f"{log_prefix}: 客户端文本发送任务结束。"); session_context._client_text_queue_ended = True

    async def _handle_input_for_activation_or_conversation(self, session_context: SessionContext, input_text: str,
                                                           input_source: str = "unknown"):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        log_prefix = f"ChatEngine [{session_context.session_id}] _handle_input_activation"
        if not input_text.strip():
            logger.debug(f"{log_prefix}: 收到空文本输入，不处理。来源: {input_source}")
            if input_source == "voice_asr" and session_context._last_vad_judgement_is_speech: session_context.update_interaction_time()
            session_context._last_vad_judgement_is_speech = False;
            return
        logger.info(f"{log_prefix}: 处理文本 '{input_text[:50]}...' (来源: {input_source})")
        if session_context.enable_prompt_activation and not session_context.is_active:
            is_activation_cmd = any(
                keyword.lower() in input_text.lower() for keyword in session_context.activation_keywords)
            if is_activation_cmd:
                session_context.is_active = True;
                session_context.update_interaction_time()
                matched_keyword = next(
                    (kw for kw in session_context.activation_keywords if kw.lower() in input_text.lower()),
                    input_text[:20])
                logger.info(f"{log_prefix}: 会话已通过指令 '{matched_keyword}' (来源: {input_source}) 激活。")
                await self._send_system_or_error_reply(session_context, session_context.activation_reply,
                                                       "system_message")
            else:
                logger.info(f"{log_prefix}: (未激活) 收到非激活指令: '{input_text[:20]}...' (来源: {input_source})")
                if input_source == "voice_asr" and session_context._last_vad_judgement_is_speech:
                    logger.debug(f"{log_prefix}: (未激活) 语音输入非激活指令，但VAD检测到语音，更新交互时间。")
                    session_context.update_interaction_time()
                if session_context.prompt_if_not_activated: await self._send_system_or_error_reply(session_context,
                                                                                                   session_context.prompt_if_not_activated,
                                                                                                   "system_message")
            session_context._last_vad_judgement_is_speech = False
        else:
            session_context.update_interaction_time();
            session_context._last_vad_judgement_is_speech = False
            await self._process_active_conversation(session_context, input_text)

    async def _process_active_conversation(self, session_context: SessionContext, text_input: str):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        log_prefix = f"ChatEngine [{session_context.session_id}] _process_active_conversation"
        logger.info(f"{log_prefix}: 处理文本: '{text_input[:50]}...'")
        llm_module: Optional[BaseLLM] = self._get_module_instance("llm")
        if not (llm_module and llm_module.is_ready):
            logger.warning(f"{log_prefix}: LLM模块不可用。")
            await self._send_system_or_error_reply(session_context, f"LLM模块当前不可用。您说的是: '{text_input}'",
                                                   "error_response");
            return
        llm_input_text_data = TextData(text=text_input, chunk_id=session_context.session_id)
        if llm_module.streaming_enabled:
            logger.info(f"{log_prefix}: LLM启用流式处理。")
            tts_input_queue = asyncio.Queue(maxsize=100);
            client_text_chunk_queue = asyncio.Queue(maxsize=100)
            for task_attr in ['tts_consumer_task', 'client_text_sender_task', 'llm_producer_task']:
                existing_task = getattr(session_context, task_attr, None)
                if existing_task and not existing_task.done(): existing_task.cancel(); await asyncio.sleep(0.01)
                setattr(session_context, task_attr, None)
            session_context.tts_consumer_task = asyncio.create_task(
                self._tts_consumer_task(session_context, tts_input_queue))
            session_context.client_text_sender_task = asyncio.create_task(
                self._client_text_sender_task(session_context, client_text_chunk_queue))
            llm_stream = llm_module.stream_chat_response(llm_input_text_data, session_context.session_id)
            session_context.llm_producer_task = asyncio.create_task(
                self._llm_to_tts_producer_task(session_context, llm_stream, tts_input_queue, client_text_chunk_queue))
            session_context.current_processing_task = session_context.llm_producer_task
            try:
                await session_context.llm_producer_task
                await client_text_chunk_queue.join();
                await tts_input_queue.join()
                logger.info(f"{log_prefix}: 所有流式处理任务已join。")
            except asyncio.CancelledError:
                logger.info(f"{log_prefix}: 主LLM处理任务被取消。")
            except Exception as e_stream_main:
                logger.error(f"{log_prefix}: 主LLM流处理中发生错误: {e_stream_main}",
                             exc_info=True); await self._send_system_or_error_reply(session_context,
                                                                                    "处理您的请求时发生内部错误。",
                                                                                    "error_response")
            finally:
                logger.info(f"{log_prefix}: 流式对话处理结束。相关任务将在会话关闭时清理。")
                tasks_to_await = [task for task in
                                  [session_context.client_text_sender_task, session_context.tts_consumer_task] if
                                  task and not task.done()]
                if tasks_to_await:
                    done, pending = await asyncio.wait(tasks_to_await, timeout=5.0, return_when=asyncio.ALL_COMPLETED)
                    for task_p in pending: task_name_p = task_p.get_name() if hasattr(task_p, 'get_name') else str(
                        task_p); logger.warning(
                        f"{log_prefix}: 任务 {task_name_p} 超时未结束，将被取消。"); task_p.cancel()
                    if pending: await asyncio.gather(*pending, return_exceptions=True)
        else:
            logger.info(f"{log_prefix}: LLM使用非流式处理。")
            llm_response_text = f"LLM模块当前不可用。您说的是: '{text_input}'"
            try:
                response_data = await llm_module.generate_complete_response(llm_input_text_data,
                                                                            session_context.session_id)
                if isinstance(response_data, TextData) and response_data.text:
                    llm_response_text = response_data.text
                elif isinstance(response_data, str) and response_data:
                    llm_response_text = response_data
                else:
                    logger.error(
                        f"{log_prefix}: LLM模块返回了非预期或空的回复: {response_data}"); llm_response_text = f"抱歉，我暂时无法理解您说的“{text_input[:30]}...”，请换个说法试试。"
            except Exception as e_llm_non_stream:
                logger.error(f"{log_prefix}: LLM非流式处理时出错: {e_llm_non_stream}",
                             exc_info=True); llm_response_text = f"抱歉，处理“{text_input[:30]}...”时遇到问题，请稍后再试。"
            await self._send_system_or_error_reply(session_context, llm_response_text, message_type="text_response")

    def _initialize_session_audio_processing_state(self, session_context: SessionContext,
                                                   vad_module: Optional[BaseVAD] = None):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        session_context._raw_audio_buffer_vad.clear()
        session_context._speech_utterance_buffer_vad.clear()
        session_context._is_currently_speaking_vad = False
        session_context._silence_since_last_speech_vad = None
        session_context._asr_triggered_for_utterance_vad = False
        session_context._last_vad_judgement_is_speech = False
        session_context._vad_produced_asr_result_this_stream = False
        if vad_module and vad_module.is_ready:
            if session_context._vad_window_size_bytes_vad is None:
                vad_adapter_name = vad_module.get_config_key()
                session_vad_adapter_config = session_context.session_config.get("modules", {}).get("vad", {}).get(
                    "config", {}).get(vad_adapter_name, {})
                global_vad_adapter_config = self.global_config.get("modules", {}).get("vad", {}).get("config", {}).get(
                    vad_adapter_name, {})
                vad_window_samples = int(
                    session_vad_adapter_config.get("window_size_samples") or global_vad_adapter_config.get(
                        "window_size_samples", 512 if vad_module.default_sample_rate == 16000 else 256))
                session_context._vad_window_size_bytes_vad = vad_window_samples * 2 * 1
            if session_context._min_silence_for_eos_ms_vad is None:
                activation_settings_session = session_context.session_config.get("activation_settings", {})
                vad_adapter_name = vad_module.get_config_key()
                vad_session_adapter_config = session_context.session_config.get("modules", {}).get("vad", {}).get(
                    "config", {}).get(vad_adapter_name, {})
                vad_global_adapter_config = self.global_config.get("modules", {}).get("vad", {}).get("config", {}).get(
                    vad_adapter_name, {})
                session_context._min_silence_for_eos_ms_vad = int(
                    activation_settings_session.get("min_silence_duration_ms_eos") or vad_session_adapter_config.get(
                        "min_silence_duration_ms_eos") or vad_global_adapter_config.get(
                        "min_silence_duration_ms_eos") or vad_session_adapter_config.get(
                        "min_silence_duration_ms") or vad_global_adapter_config.get(
                        "min_silence_duration_ms") or self.default_min_silence_ms_for_eos)
                # 新增：初始化最大语音段时长
                session_context._max_speech_segment_ms_vad = int(
                    vad_session_adapter_config.get("max_speech_segment_duration_ms") or
                    vad_global_adapter_config.get("max_speech_segment_duration_ms") or
                    self.default_max_speech_segment_ms
                )
        logger.info(
            f"ChatEngine [{session_context.session_id}]: 会话音频处理状态已初始化/重置。VAD窗口大小: {session_context._vad_window_size_bytes_vad or 'N/A'} 字节, 语句结束静默: {session_context._min_silence_for_eos_ms_vad or 'N/A'} ms, 最大语音段: {getattr(session_context, '_max_speech_segment_ms_vad', 'N/A')} ms.")

    async def _trigger_asr_for_utterance(self, session_context: SessionContext, audio_format_enum: AudioFormat,
                                         sample_rate: int, channels: int, sample_width: int, reason: str = "unknown"):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致，确保打印ASR结果)
        log_prefix = f"ChatEngine [{session_context.session_id}] _trigger_asr"
        asr_module: Optional[BaseASR] = self._get_module_instance("asr")
        speech_buffer = session_context._speech_utterance_buffer_vad
        if not speech_buffer:
            logger.debug(f"{log_prefix}: 尝试触发ASR (原因: {reason})，但语音段缓冲区为空。")
            if reason == "silence_timeout" or reason == "max_segment_length":  # 如果是这两种原因，重置说话状态
                session_context._is_currently_speaking_vad = False
                session_context._silence_since_last_speech_vad = None
            return
        complete_utterance_bytes = b"".join(list(speech_buffer));
        speech_buffer.clear()
        logger.info(f"{log_prefix}: 准备将 {len(complete_utterance_bytes)} 字节的完整语音段送往ASR (原因: {reason})。")
        if complete_utterance_bytes:
            if asr_module and asr_module.is_ready:
                utterance_ad = AudioData(data=complete_utterance_bytes, format=audio_format_enum,
                                         sample_rate=sample_rate, channels=channels, sample_width=sample_width,
                                         is_final=True, session_id=session_context.session_id,
                                         chunk_id=f"{session_context.session_id}_asr_utt_{time.time_ns()}")
                try:
                    asr_res = await asr_module.recognize_audio_block(utterance_ad, session_context.session_id)
                    if asr_res:
                        logger.info(f"{log_prefix}: ASR识别文本: '{asr_res.text[:100]}...'")  # 打印ASR结果
                        await self._handle_input_for_activation_or_conversation(session_context, asr_res.text,
                                                                                "voice_asr")
                        session_context._vad_produced_asr_result_this_stream = True
                except Exception as e_asr:
                    logger.error(f"{log_prefix}: ASR处理完整语句时出错: {e_asr}", exc_info=True)
                    session_context.update_interaction_time()
            else:
                logger.warning(f"{log_prefix}: 有语音段但ASR模块不可用。")
                session_context.update_interaction_time()
        else:
            logger.debug(f"{log_prefix}: 语句结束，但语音段缓冲区在取出后为空。")
        session_context._is_currently_speaking_vad = False
        session_context._silence_since_last_speech_vad = None
        session_context._asr_triggered_for_utterance_vad = False

    async def _handle_audio_chunk(self, session_context: SessionContext, audio_chunk_bytes: bytes,
                                  client_message_details: Dict[str, Any]):
        log_prefix = f"ChatEngine [{session_context.session_id}] _handle_audio_chunk"
        vad_module: Optional[BaseVAD] = self._get_module_instance("vad")
        if not (vad_module and vad_module.is_ready): logger.warning(f"{log_prefix}: VAD模块不可用。"); return

        self._initialize_session_audio_processing_state(session_context, vad_module)  # 确保状态已初始化

        audio_format_str = client_message_details.get("audio_format", "pcm");
        audio_format_enum = AudioFormat.PCM
        try:
            audio_format_enum = AudioFormat(audio_format_str.lower())
        except ValueError:
            pass
        sample_rate = int(client_message_details.get("sample_rate") or vad_module.default_sample_rate)
        channels = int(client_message_details.get("channels") or 1);
        sample_width = int(client_message_details.get("sample_width") or 2)

        session_context._raw_audio_buffer_vad.append(audio_chunk_bytes)
        current_raw_buffer_length = sum(len(b) for b in session_context._raw_audio_buffer_vad)
        vad_window_size_bytes = session_context._vad_window_size_bytes_vad

        # 获取会话特定的最大语音段长度（字节）
        max_speech_segment_ms = session_context._max_speech_segment_ms_vad
        max_speech_segment_bytes = (
                                               max_speech_segment_ms / 1000.0) * sample_rate * channels * sample_width if max_speech_segment_ms else float(
            'inf')

        while current_raw_buffer_length >= vad_window_size_bytes:
            vad_window_data_bytes = b'';
            accumulated_length = 0;
            temp_window_list = []
            while session_context._raw_audio_buffer_vad and accumulated_length < vad_window_size_bytes:
                front_chunk = session_context._raw_audio_buffer_vad.popleft();
                needed = vad_window_size_bytes - accumulated_length
                if len(front_chunk) <= needed:
                    temp_window_list.append(front_chunk); accumulated_length += len(front_chunk)
                else:
                    temp_window_list.append(front_chunk[:needed]); session_context._raw_audio_buffer_vad.appendleft(
                        front_chunk[needed:]); accumulated_length += needed
            vad_window_data_bytes = b"".join(temp_window_list);
            current_raw_buffer_length = sum(len(b) for b in session_context._raw_audio_buffer_vad)
            if not vad_window_data_bytes: continue

            vad_input_ad = AudioData(data=vad_window_data_bytes, format=audio_format_enum, sample_rate=sample_rate,
                                     channels=channels, sample_width=sample_width, is_final=False,
                                     session_id=session_context.session_id)
            is_speech = False
            try:
                is_speech = await vad_module.is_speech_present(
                    vad_input_ad); session_context._last_vad_judgement_is_speech = is_speech
            except Exception as e_vad:
                logger.error(f"{log_prefix}: VAD判断出错: {e_vad}", exc_info=True)

            if is_speech:
                session_context._speech_utterance_buffer_vad.append(vad_window_data_bytes);
                session_context._is_currently_speaking_vad = True
                session_context._silence_since_last_speech_vad = None;
                session_context._asr_triggered_for_utterance_vad = False
                session_context.update_interaction_time()

                # 检查是否达到最大语音段长度
                current_utterance_len_bytes = sum(len(b) for b in session_context._speech_utterance_buffer_vad)
                if current_utterance_len_bytes >= max_speech_segment_bytes and \
                        not session_context._asr_triggered_for_utterance_vad and \
                        session_context._speech_utterance_buffer_vad:
                    logger.info(
                        f"{log_prefix}: 语音段达到最大长度 ({current_utterance_len_bytes} >= {max_speech_segment_bytes:.0f} bytes)，触发ASR。")
                    session_context._asr_triggered_for_utterance_vad = True
                    await self._trigger_asr_for_utterance(session_context, audio_format_enum, sample_rate, channels,
                                                          sample_width, reason="max_segment_length")
            else:  # 当前VAD窗口无语音
                if session_context._is_currently_speaking_vad:  # 如果之前在说话
                    if session_context._silence_since_last_speech_vad is None: session_context._silence_since_last_speech_vad = time.time()
                    current_silence_ms = (time.time() - session_context._silence_since_last_speech_vad) * 1000
                    min_silence_eos_ms = session_context._min_silence_for_eos_ms_vad
                    if current_silence_ms >= min_silence_eos_ms and not session_context._asr_triggered_for_utterance_vad and session_context._speech_utterance_buffer_vad:
                        logger.info(
                            f"{log_prefix}: 语音后静默达到 {current_silence_ms:.0f}ms (阈值 {min_silence_eos_ms}ms)，触发ASR。")
                        session_context._asr_triggered_for_utterance_vad = True
                        await self._trigger_asr_for_utterance(session_context, audio_format_enum, sample_rate, channels,
                                                              sample_width)

    async def _handle_audio_end(self, session_context: SessionContext, client_message_details: Dict[str, Any]):
        log_prefix = f"ChatEngine [{session_context.session_id}] _handle_audio_end"
        logger.info(f"{log_prefix}: 收到音频流结束信号。详情: {client_message_details}")
        vad_module: Optional[BaseVAD] = self._get_module_instance("vad");
        asr_module: Optional[BaseASR] = self._get_module_instance("asr")
        if not (vad_module and vad_module.is_ready):
            logger.warning(f"{log_prefix}: VAD模块不可用，无法在音频结束时处理剩余缓冲。")
            if hasattr(session_context,
                       '_speech_utterance_buffer_vad') and session_context._speech_utterance_buffer_vad:  # 尝试处理已累积的语音
                audio_format_str = client_message_details.get("audio_format", "pcm");
                audio_format_enum = AudioFormat.PCM
                try:
                    audio_format_enum = AudioFormat(audio_format_str.lower())
                except ValueError:
                    pass
                sr = int(client_message_details.get("sample_rate") or (
                    asr_module.expected_sample_rate if asr_module else 16000))
                ch = int(client_message_details.get("channels") or (asr_module.expected_channels if asr_module else 1))
                sw = int(client_message_details.get("sample_width") or (
                    asr_module.expected_sample_width if asr_module else 2))
                await self._trigger_asr_for_utterance(session_context, audio_format_enum, sr, ch, sw,
                                                      reason="audio_end_no_vad")
            self._reset_session_audio_processing_state(session_context);
            return

        self._initialize_session_audio_processing_state(session_context, vad_module)

        if session_context._raw_audio_buffer_vad:
            remaining_raw_bytes = b"".join(list(session_context._raw_audio_buffer_vad));
            session_context._raw_audio_buffer_vad.clear()
            if remaining_raw_bytes:
                logger.debug(f"{log_prefix}: 音频结束，最后处理原始缓冲区中 {len(remaining_raw_bytes)} 字节。")
                await self._handle_audio_chunk(session_context, remaining_raw_bytes, client_message_details)

        if session_context._speech_utterance_buffer_vad or session_context._is_currently_speaking_vad:
            if not session_context._asr_triggered_for_utterance_vad or session_context._speech_utterance_buffer_vad:
                logger.info(f"{log_prefix}: 音频流结束，强制处理当前累积的语音段（如果有）。")
                audio_format_str = client_message_details.get("audio_format", "pcm");
                audio_format_enum = AudioFormat.PCM
                try:
                    audio_format_enum = AudioFormat(audio_format_str.lower())
                except ValueError:
                    pass
                asr_expected_sr = asr_module.expected_sample_rate if asr_module else 16000;
                asr_expected_ch = asr_module.expected_channels if asr_module else 1;
                asr_expected_sw = asr_module.expected_sample_width if asr_module else 2
                sample_rate = int(client_message_details.get("sample_rate") or asr_expected_sr);
                channels = int(client_message_details.get("channels") or asr_expected_ch);
                sample_width = int(client_message_details.get("sample_width") or asr_expected_sw)
                await self._trigger_asr_for_utterance(session_context, audio_format_enum, sample_rate, channels,
                                                      sample_width, reason="audio_stream_end")
            else:
                logger.debug(f"{log_prefix}: 音频流结束，但当前语句ASR已触发或缓冲区为空。")
        else:
            logger.debug(f"{log_prefix}: 音频流结束，当前语句缓冲区已为空。")
            if session_context.is_active and not session_context._vad_produced_asr_result_this_stream:
                await self._send_system_or_error_reply(session_context, "我好像没有听到您说什么，请再说一遍好吗？",
                                                       "system_message")

        self._reset_session_audio_processing_state(session_context)
        if vad_module and vad_module.is_ready and hasattr(vad_module, 'reset_state'): await vad_module.reset_state()
        if asr_module and asr_module.is_ready and hasattr(asr_module,
                                                          'reset_state'): await asr_module.reset_state()  # type: ignore

    def _reset_session_audio_processing_state(self, session_context: SessionContext):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        log_prefix = f"ChatEngine [{session_context.session_id}] _reset_audio_state"
        session_context._raw_audio_buffer_vad.clear()
        session_context._speech_utterance_buffer_vad.clear()
        session_context._is_currently_speaking_vad = False
        session_context._silence_since_last_speech_vad = None
        session_context._asr_triggered_for_utterance_vad = False
        session_context._last_vad_judgement_is_speech = False
        session_context._vad_produced_asr_result_this_stream = False
        logger.debug(f"{log_prefix}: 会话音频处理状态已重置。")

    async def _process_message_for_session(self, session_context: SessionContext, payload: Any, origin_type: str,
                                           client_message_details: Optional[Dict[str, Any]] = None):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        session_ctx_token = current_user_session_var.set(session_context);
        client_message_details = client_message_details or {}
        if origin_type in ["audio_chunk", "audio_end"]:
            vad_module: Optional[BaseVAD] = self._get_module_instance("vad")
            if vad_module and vad_module.is_ready: self._initialize_session_audio_processing_state(session_context,
                                                                                                   vad_module)
        try:
            if origin_type == "text" or origin_type == "raw_text":
                text_input = str(payload);
                logger.info(f"[{session_context.session_id}] 收到文本输入 ('{origin_type}'): '{text_input[:50]}...'")
                await self._handle_input_for_activation_or_conversation(session_context, text_input, "text_direct")
            elif origin_type == "audio_chunk":
                if isinstance(payload, bytes):
                    await self._handle_audio_chunk(session_context, payload, client_message_details)
                else:
                    logger.warning(
                        f"[{session_context.session_id}] 收到类型为 'audio_chunk' 的请求，但载荷不是字节: {type(payload)}"); await session_context.send_message_to_client(
                        {"type": "error", "message": "无效的音频数据格式。"})
            elif origin_type == "audio_end":
                await self._handle_audio_end(session_context, client_message_details)
            else:
                logger.warning(
                    f"[{session_context.session_id}] ChatEngine 收到未明确处理的 origin_type: '{origin_type}'。"); await session_context.send_message_to_client(
                    {"type": "error", "message": f"服务器无法处理类型为 '{origin_type}' 的请求。"})
        except Exception as e:
            logger.error(f"[{session_context.session_id}] 处理消息时发生严重错误: {e}", exc_info=True)
            try:
                await session_context.send_message_to_client({"type": "error", "message": "服务器内部处理错误。"})
            except Exception as e_send:
                logger.error(f"[{session_context.session_id}] 发送错误消息给客户端失败: {e_send}")
        finally:
            current_user_session_var.reset(session_ctx_token)

    async def handle_websocket_connection(self, websocket: Any, path: Optional[str] = None):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        client_address = websocket.remote_address;
        session_context: Optional[SessionContext] = None;
        assigned_session_id: Optional[str] = None
        logger.info(f"客户端 {client_address} 尝试通过路径 '{path}' 连接...")
        try:
            initial_message_raw = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            if not isinstance(initial_message_raw, str): logger.warning(
                f"客户端 {client_address} 的第一条消息不是字符串，关闭连接。"); await websocket.close(code=1003,
                                                                                                    reason="Invalid first message type"); return
            try:
                initial_data = json.loads(initial_message_raw)
            except json.JSONDecodeError:
                logger.warning(f"客户端 {client_address} 的第一条消息不是有效的JSON，关闭连接。"); await websocket.close(
                    code=1003, reason="Invalid JSON"); return
            if initial_data.get("type") != "init_session": logger.warning(
                f"客户端 {client_address} 的第一条消息类型不是 'init_session'，关闭连接。"); await websocket.close(
                code=1002, reason="First message must be 'init_session'."); return
            external_session_id_from_client = initial_data.get("session_id");
            user_specific_config_data = initial_data.get("config", {});
            client_audio_params_from_init = user_specific_config_data.get("client_audio_params", {})
            logger.info(
                f"客户端 {client_address} 发送初始化请求，session_id: {external_session_id_from_client}, 用户配置: {user_specific_config_data}")
            if external_session_id_from_client:
                try:
                    uuid.UUID(external_session_id_from_client)
                except ValueError:
                    external_session_id_from_client = str(uuid.uuid4())
                session_context = self.active_sessions.get(external_session_id_from_client)
                if session_context:
                    logger.info(f"客户端 {client_address} 重新连接到现有会话: {external_session_id_from_client}")
                    session_context.websocket = websocket;
                    session_context.user_specific_config_overrides.update(user_specific_config_data);
                    session_context.session_config = session_context._merge_configs();
                    session_context.update_websocket_activity_time();
                    assigned_session_id = external_session_id_from_client
                    vad_module_instance = self._get_module_instance("vad")
                    if vad_module_instance and vad_module_instance.is_ready: self._initialize_session_audio_processing_state(
                        session_context, vad_module_instance)
                else:
                    logger.info(
                        f"为客户端 {client_address} 使用其提供的ID创建新会话: {external_session_id_from_client}")
                    session_context = SessionContext(session_id=external_session_id_from_client, websocket=websocket,
                                                     global_config=self.global_config,
                                                     user_specific_config=user_specific_config_data)
                    self.active_sessions[external_session_id_from_client] = session_context;
                    assigned_session_id = external_session_id_from_client
                    vad_module_instance = self._get_module_instance("vad")
                    if vad_module_instance and vad_module_instance.is_ready: self._initialize_session_audio_processing_state(
                        session_context, vad_module_instance)
            else:
                new_session_id = str(uuid.uuid4());
                logger.info(f"为客户端 {client_address} 生成新的会话ID: {new_session_id}")
                session_context = SessionContext(session_id=new_session_id, websocket=websocket,
                                                 global_config=self.global_config,
                                                 user_specific_config=user_specific_config_data)
                self.active_sessions[new_session_id] = session_context;
                assigned_session_id = new_session_id
                vad_module_instance = self._get_module_instance("vad")
                if vad_module_instance and vad_module_instance.is_ready: self._initialize_session_audio_processing_state(
                    session_context, vad_module_instance)
            if not session_context or not assigned_session_id: logger.error(
                f"为客户端 {client_address} 创建/获取会话上下文失败。"); await websocket.close(code=1011,
                                                                                              reason="服务器会话处理错误"); return
            await websocket.send(json.dumps(
                {"type": "session_established", "session_id": assigned_session_id, "message": "会话已成功建立。",
                 "activation_enabled": session_context.enable_prompt_activation,
                 "is_active_initially": session_context.is_active}))
            logger.info(
                f"[{assigned_session_id}] 会话建立确认已发送。激活启用: {session_context.enable_prompt_activation}, 初始激活: {session_context.is_active}")
            async for message_raw in websocket:
                session_context.update_websocket_activity_time();
                internal_origin_type = "unknown_client_type";
                payload_to_process: Any = None;
                message_details: Dict[str, Any] = {}
                if client_audio_params_from_init: message_details.update(client_audio_params_from_init)
                if isinstance(message_raw, str):
                    try:
                        data = json.loads(message_raw);
                        client_message_type = data.get("type");
                        payload_to_process = data.get("content", data.get("text"))
                        if client_message_type in ["text_message", "text_input_from_client", "chat_message"]:
                            internal_origin_type = "text"
                        elif client_message_type == "audio_stream_end_signal":
                            internal_origin_type = "audio_end"; payload_to_process = None
                        elif client_message_type == "audio_input":
                            internal_origin_type = "audio_chunk";
                            payload_to_process = base64.b64decode(data.get("data", "")) if "data" in data else None
                            message_details["audio_format"] = data.get("audio_format",
                                                                       message_details.get("audio_format", "pcm"));
                            message_details["sample_rate"] = data.get("sample_rate",
                                                                      message_details.get("sample_rate"));
                            message_details["channels"] = data.get("channels", message_details.get("channels"));
                            message_details["sample_width"] = data.get("sample_width",
                                                                       message_details.get("sample_width"))
                        else:
                            logger.warning(
                                f"[{assigned_session_id}] 收到未映射的客户端JSON消息类型: {client_message_type}, data: {data}"); payload_to_process = data; internal_origin_type = client_message_type or "unknown_json_type"
                        if payload_to_process is not None or internal_origin_type == "audio_end":
                            await self._process_message_for_session(session_context, payload_to_process,
                                                                    internal_origin_type, message_details)
                        elif client_message_type and not payload_to_process and internal_origin_type == "text":
                            logger.warning(
                                f"[{assigned_session_id}] 文本类型消息 '{client_message_type}' 缺少有效载荷。"); await session_context.send_message_to_client(
                                {"type": "error",
                                 "message": f"消息类型 '{client_message_type}' 需要 'text' 或 'content' 字段。"})
                    except json.JSONDecodeError:
                        logger.warning(
                            f"[{assigned_session_id}] 收到的文本消息非JSON格式: {message_raw[:100]}。将作为raw_text处理。"); await self._process_message_for_session(
                            session_context, message_raw, "raw_text", message_details)
                elif isinstance(message_raw, bytes):
                    if not all(k in message_details for k in ["sample_rate", "channels", "sample_width"]):
                        asr_module_instance = self._get_module_instance("asr");
                        asr_session_config = {};
                        asr_global_config = {}
                        if asr_module_instance and asr_module_instance.enabled_adapter_name:
                            asr_session_config = session_context.session_config.get("modules", {}).get("asr", {}).get(
                                "config", {}).get(asr_module_instance.enabled_adapter_name, {})
                            asr_global_config = self.global_config.get("modules", {}).get("asr", {}).get("config",
                                                                                                         {}).get(
                                asr_module_instance.enabled_adapter_name, {})
                        global_default_pcm_params = self.global_config.get("default_pcm_params", {})
                        message_details.setdefault("audio_format", "pcm");
                        message_details.setdefault("sample_rate",
                                                   asr_session_config.get("sample_rate") or asr_global_config.get(
                                                       "sample_rate") or global_default_pcm_params.get("sample_rate",
                                                                                                       16000));
                        message_details.setdefault("channels",
                                                   asr_session_config.get("channels") or asr_global_config.get(
                                                       "channels") or global_default_pcm_params.get("channels", 1));
                        message_details.setdefault("sample_width",
                                                   asr_session_config.get("sample_width") or asr_global_config.get(
                                                       "sample_width") or global_default_pcm_params.get("sample_width",
                                                                                                        2))
                    await self._process_message_for_session(session_context, message_raw, "audio_chunk",
                                                            message_details)
        except websockets.exceptions.ConnectionClosed as e_closed:
            logger.info(
                f"客户端 {client_address} (会话: {assigned_session_id or 'N/A'}) 连接已关闭 (Code: {e_closed.code if hasattr(e_closed, 'code') else e_closed.rcvd.code if hasattr(e_closed, 'rcvd') and e_closed.rcvd else 'N/A'}, Reason: '{e_closed.reason if hasattr(e_closed, 'reason') else e_closed.rcvd.reason if hasattr(e_closed, 'rcvd') and e_closed.rcvd else 'N/A'}').")
        except asyncio.TimeoutError:
            logger.warning(
                f"客户端 {client_address} 在规定时间内未发送 'init_session' 消息，连接已关闭。"); await websocket.close(
                code=1008, reason="Initialization timeout.")
        except Exception as e_conn:
            logger.error(f"处理客户端 {client_address} (会话: {assigned_session_id or 'N/A'}) 连接时发生错误: {e_conn}",
                         exc_info=True); await websocket.close(code=1011, reason="服务器内部错误")
        finally:
            if assigned_session_id and assigned_session_id in self.active_sessions:
                logger.info(f"正在清理因连接结束或错误而终止的会话 '{assigned_session_id}'...")
                session_to_close = self.active_sessions.pop(assigned_session_id, None)
                if session_to_close: self._reset_session_audio_processing_state(
                    session_to_close); await session_to_close.close(reason="WebSocket 连接结束或处理错误")

    async def _periodic_activation_timeout_check(self):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        logger.info("ChatEngine 的激活状态超时检查任务已启动。")
        while not self.is_shutting_down:
            await asyncio.sleep(DEFAULT_ACTIVATION_TIMEOUT_CHECK_INTERVAL_SECONDS);
            now = time.time()
            if self.is_shutting_down: break
            for session_ctx in list(self.active_sessions.values()):
                if session_ctx.enable_prompt_activation and session_ctx.is_active and \
                        (now - session_ctx.last_interaction_time) > session_ctx.activation_timeout_seconds:
                    session_ctx.is_active = False
                    logger.info(f"[{session_ctx.session_id}] 会话因无交互超时自动取消激活。")
                    await self._send_system_or_error_reply(session_ctx, session_ctx.deactivation_reply,
                                                           "system_message")
        logger.info("ChatEngine 的激活状态超时检查任务已停止。")

    async def _periodic_session_cleanup(self):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        logger.info("ChatEngine 的会话超时清理任务已启动。")
        while not self.is_shutting_down:
            await asyncio.sleep(max(30, self.session_timeout_seconds // 10) if self.active_sessions else 60)
            if self.is_shutting_down: break; now = time.time(); expired_session_ids: List[str] = []
            for sid, session_ctx in list(self.active_sessions.items()):
                if (now - session_ctx.last_websocket_activity_time) > self.session_timeout_seconds:
                    expired_session_ids.append(sid)
            if expired_session_ids:
                logger.info(f"发现 {len(expired_session_ids)} 个整体超时会话，正在清理: {expired_session_ids}")
                for sid_to_delete in expired_session_ids:
                    if sid_to_delete in self.active_sessions:
                        session_to_close = self.active_sessions.pop(sid_to_delete, None)
                        if session_to_close: self._reset_session_audio_processing_state(
                            session_to_close); await session_to_close.close(reason="WebSocket 连接超时")
        logger.info("ChatEngine 的会话超时清理任务已停止。")

    async def shutdown(self):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        logger.info("全局 ChatEngine 正在关闭...")
        self.is_shutting_down = True
        tasks_to_cancel = [self._activation_check_task, self._cleanup_task]
        for task in tasks_to_cancel:
            if task and not task.done():
                task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
                task.cancel();
                await asyncio.sleep(0.1)
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.CancelledError:
                    logger.info(f"任务 {task_name} 已成功取消。")
                except asyncio.TimeoutError:
                    logger.warning(f"任务 {task_name} 取消超时。")
        self._activation_check_task = None;
        self._cleanup_task = None
        closing_session_tasks = []
        for session_id in list(self.active_sessions.keys()):
            session_ctx = self.active_sessions.pop(session_id, None)
            if session_ctx: self._reset_session_audio_processing_state(session_ctx); closing_session_tasks.append(
                session_ctx.close("ChatEngine 关闭"))
        if closing_session_tasks: await asyncio.gather(*closing_session_tasks, return_exceptions=True)
        self.active_sessions.clear()
        if self.module_manager: await self.module_manager.shutdown_modules()
        logger.info("全局 ChatEngine 关闭完成。")

    @classmethod
    async def create_and_run(cls, config_file_path: str, host: str, port: int):
        # ... (与 chat_engine_activation_logic_refined_v1 版本一致)
        logger.info(f"ChatEngine 服务正在尝试从配置文件 '{config_file_path}' 启动...")
        if not os.path.exists(config_file_path): logger.critical(f"错误: 配置文件 '{config_file_path}' 未找到。"); return
        try:
            config_loader = ConfigLoader();
            global_app_config = config_loader.load_config(config_file_path)
            if not global_app_config: logger.critical(f"错误: 从 '{config_file_path}' 加载配置失败。"); return
            engine = cls(config=global_app_config, loop=asyncio.get_event_loop())
            await engine.initialize();
            logger.info("全局 ChatEngine 初始化成功。")
            server_instance = None
            try:
                ws_max_size = global_app_config.get("websocket_server", {}).get("websocket_max_message_size",
                                                                                1024 * 1024)
                logger.info(f"准备在 ws://{host}:{port} 上启动 WebSocket 服务器 (max_size={ws_max_size})...")
                server_instance = await websockets.serve(engine.handle_websocket_connection, host, port,
                                                         max_size=ws_max_size)  # type: ignore
                logger.info(f"WebSocket 服务器已在 ws://{host}:{port} 上启动，由 ChatEngine 管理。")
                await asyncio.Future()
            except OSError as e_os:
                logger.critical(f"启动 WebSocket 服务器失败: {e_os}", exc_info=True)
            except Exception as e_serve:
                logger.critical(f"运行 WebSocket 服务器时发生未知错误: {e_serve}", exc_info=True)
            finally:
                if server_instance: logger.info(
                    "正在关闭 WebSocket 服务器..."); server_instance.close(); await server_instance.wait_closed()
                logger.info("正在关闭 ChatEngine 服务...");
                await engine.shutdown();
                logger.info("ChatEngine 服务已关闭。")
        except Exception as e_outer:
            logger.critical(f"ChatEngine 服务启动或运行期间发生顶层错误: {e_outer}", exc_info=True)

