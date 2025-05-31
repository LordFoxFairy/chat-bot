# core_framework/chat_engine.py
import base64
import logging
from typing import Dict, Any, Optional, Union

from core_framework.exceptions import ModuleProcessingError  # 假设您有这个异常类
from data_models.audio_data import AudioData  # 使用更新后的 AudioData
from data_models.text_data import TextData
from .module_manager import ModuleManager

# 模块基类和具体模块的类型提示 (概念性，实际实现可能不同)
# from modules.base_asr import BaseASR
# from modules.base_llm import BaseLLM
# from modules.base_tts import BaseTTS

logger = logging.getLogger(__name__)


class ChatEngine:
    """
    ChatEngine负责协调ASR, LLM, TTS模块来处理用户输入并生成回复。
    模块的 process 方法应直接返回数据对象或None，或抛出 ModuleProcessingError。
    """

    def __init__(self,module_manager: ModuleManager):
        self.module_manager = module_manager
        self.asr_module: Optional[Any] = None  # 应为 BaseASR 类型
        self.llm_module: Optional[Any] = None  # 应为 BaseLLM 类型
        self.tts_module: Optional[Any] = None  # 应为 BaseTTS 类型
        self._load_modules()

    def _load_modules(self):
        """
        从ModuleManager加载ASR, LLM, TTS模块。
        模块名称（如'asr', 'llm', 'tts'）应与配置文件中的定义一致。
        """
        try:
            self.asr_module = self.module_manager.get_module("asr")
            if not self.asr_module:
                logger.warning("ASR模块 ('asr') 未找到或加载。音频输入将无法处理。")

            self.llm_module = self.module_manager.get_module("llm")
            if not self.llm_module:
                logger.error("LLM模块 ('llm') 未找到或加载。ChatEngine 无法在没有LLM的情况下运行。")
                raise ValueError("LLM模块是必需的，但未能加载。")

            self.tts_module = self.module_manager.get_module("tts")
            if not self.tts_module:
                logger.warning("TTS模块 ('tts') 未找到或加载。回复将仅为文本。")

            logger.info("ChatEngine模块加载状态: ASR=%s, LLM=%s, TTS=%s",
                        "已启用" if self.asr_module else "已禁用",
                        "已启用" if self.llm_module else "已禁用",
                        "已启用" if self.tts_module else "已禁用")

        except Exception as e:
            logger.exception(f"ChatEngine加载模块时出错: {e}")
            raise

    async def process_message(self, session_id: str,
                              input_payload: Union[AudioData, TextData, str],
                              input_origin_type: str) -> Dict[str, Any]:
        """
        处理传入的消息（音频或文本）。
        :param session_id: 当前会话的ID。
        :param input_payload: 输入数据。对于音频是AudioData对象，对于文本是TextData对象或str。
        :param input_origin_type: 原始输入类型， "audio" 或 "text"。
        :return: 一个包含 'session_id', 'text', 'audio_b64' (可选), 'error' (可选) 的字典。
        """
        logger.debug(f"ChatEngine开始处理会话 {session_id} 的消息，原始类型: {input_origin_type}")

        if not self.llm_module:
            logger.error(f"会话 {session_id}: LLM模块不可用，无法处理消息。")
            return {"session_id": session_id, "text": "抱歉，语言模型服务当前不可用。", "audio_b64": None,
                    "error": "LLM_SERVICE_UNAVAILABLE"}

        text_for_llm_input: Optional[TextData] = None

        try:
            # 1. ASR处理 (如果原始输入是音频)
            if input_origin_type == "audio":
                if not self.asr_module:
                    logger.warning(f"会话 {session_id}: 收到音频输入，但ASR模块不可用。")
                    return {"session_id": session_id, "text": "抱歉，语音识别服务当前不可用。", "audio_b64": None,
                            "error": "ASR_SERVICE_UNAVAILABLE"}
                if not isinstance(input_payload, AudioData):
                    logger.error(
                        f"会话 {session_id}: 音频处理的输入数据类型无效: {type(input_payload)}，期望 AudioData。")
                    return {"session_id": session_id, "text": "抱歉，音频数据格式不正确。", "audio_b64": None,
                            "error": "INVALID_AUDIO_DATA_FORMAT"}

                logger.debug(f"会话 {session_id}: 将音频数据发送到ASR模块...")
                try:
                    # 假设ASR模块的process方法签名是: process(self, audio_input: AudioData, session_id: str) -> Optional[TextData]
                    recognized_text_data: Optional[TextData] = await self.asr_module.process(input_payload, session_id)
                except ModuleProcessingError as e:
                    logger.error(f"会话 {session_id}: ASR模块处理时发生错误: {e}", exc_info=True)
                    return {"session_id": session_id, "text": "抱歉，语音识别过程中发生错误。", "audio_b64": None,
                            "error": f"ASR_PROCESSING_ERROR: {e}"}
                except Exception as e:
                    logger.error(f"会话 {session_id}: 调用ASR模块时发生未知异常: {e}", exc_info=True)
                    return {"session_id": session_id, "text": "抱歉，语音识别服务出现意外问题。", "audio_b64": None,
                            "error": f"ASR_UNEXPECTED_ERROR: {e}"}

                if recognized_text_data and recognized_text_data.text.strip():
                    text_for_llm_input = recognized_text_data
                    logger.info(f"会话 {session_id}: ASR识别结果: '{text_for_llm_input.text}'")
                else:
                    logger.warning(f"会话 {session_id}: ASR模块未返回有效文本或返回空文本。")
                    # 根据策略，可以是错误，也可以是“未听到”类型的回复
                    return {"session_id": session_id, "text": "抱歉，我没有听清楚您说什么。", "audio_b64": None,
                            "error": "ASR_NO_RECOGNITION"}

            elif input_origin_type == "text":
                if isinstance(input_payload, TextData):
                    text_for_llm_input = input_payload
                elif isinstance(input_payload, str):
                    text_for_llm_input = TextData(text=input_payload, segment_id=session_id)  # 包装成TextData
                else:
                    logger.error(
                        f"会话 {session_id}: 文本处理的输入数据类型无效: {type(input_payload)}，期望 TextData 或 str。")
                    return {"session_id": session_id, "text": "抱歉，文本消息格式不正确。", "audio_b64": None,
                            "error": "INVALID_TEXT_INPUT_FORMAT"}
                logger.info(f"会话 {session_id}: LLM的文本输入: '{text_for_llm_input.text}'")
            else:
                logger.error(f"会话 {session_id}: 不支持的原始输入类型: {input_origin_type}")
                return {"session_id": session_id, "text": "抱歉，不支持的消息类型。", "audio_b64": None,
                        "error": f"UNSUPPORTED_INPUT_TYPE: {input_origin_type}"}

            if not text_for_llm_input or not text_for_llm_input.text.strip():
                logger.warning(f"会话 {session_id}: 经过ASR/文本预处理后，没有有效的文本送给LLM。")
                return {"session_id": session_id, "text": "抱歉，我没有接收到任何有效的文本内容进行处理。",
                        "audio_b64": None, "error": "NO_VALID_TEXT_FOR_LLM"}

            # 2. LLM处理
            # 确保 text_for_llm_input 的 metadata 中有 role 信息，如果LLM模块需要
            if "role" not in text_for_llm_input.metadata:
                text_for_llm_input.metadata["role"] = "user"  # 默认设为 user

            logger.debug(f"会话 {session_id}: 将文本 '{text_for_llm_input.text[:50]}...' 发送到LLM模块...")
            try:
                # 假设LLM模块的process方法签名是: process(self, text_input: TextData, session_id: str) -> Optional[TextData]
                llm_response_text_data: Optional[TextData] = await self.llm_module.process(text_for_llm_input,
                                                                                           session_id)
            except ModuleProcessingError as e:
                logger.error(f"会话 {session_id}: LLM模块处理时发生错误: {e}", exc_info=True)
                return {"session_id": session_id, "text": "抱歉，思考回复时发生错误。", "audio_b64": None,
                        "error": f"LLM_PROCESSING_ERROR: {e}"}
            except Exception as e:
                logger.error(f"会话 {session_id}: 调用LLM模块时发生未知异常: {e}", exc_info=True)
                return {"session_id": session_id, "text": "抱歉，语言模型服务出现意外问题。", "audio_b64": None,
                        "error": f"LLM_UNEXPECTED_ERROR: {e}"}

            if not llm_response_text_data or not llm_response_text_data.text.strip():
                logger.warning(f"会话 {session_id}: LLM模块未返回有效文本或返回空文本。")
                return {"session_id": session_id, "text": "我暂时不知道该如何回复。", "audio_b64": None,
                        "error": "LLM_EMPTY_RESPONSE"}

            final_response_text = llm_response_text_data.text
            logger.info(f"会话 {session_id}: LLM回复: '{final_response_text[:50]}...'")

            # 3. TTS处理 (如果TTS模块可用且LLM有回复)
            tts_audio_b64_output: Optional[str] = None
            if self.tts_module and final_response_text:
                # 准备给TTS的TextData，可以从LLM的输出获取或新建
                text_for_tts = TextData(text=final_response_text, segment_id=session_id, metadata={"role": "assistant"})

                logger.debug(f"会话 {session_id}: 将文本 '{text_for_tts.text[:50]}...' 发送到TTS模块...")
                try:
                    # 假设TTS模块的process方法签名是: process(self, text_input: TextData, session_id: str) -> Optional[AudioData]
                    tts_output_audio_data: Optional[AudioData] = await self.tts_module.process(text_for_tts, session_id)
                except ModuleProcessingError as e:
                    logger.error(f"会话 {session_id}: TTS模块处理时发生错误: {e}", exc_info=True)
                    # TTS失败不应阻止文本回复，仅记录警告
                    logger.warning(f"会话 {session_id}: TTS处理失败，将仅发送文本回复。错误: {e}")
                except Exception as e:
                    logger.error(f"会话 {session_id}: 调用TTS模块时发生未知异常: {e}", exc_info=True)
                    logger.warning(f"会话 {session_id}: TTS服务出现意外问题，将仅发送文本回复。错误: {e}")
                else:
                    if tts_output_audio_data and tts_output_audio_data.audio:
                        # 检查音频类型，理想情况下TTS应输出 "audio/pcm" 或可以直接编码的类型
                        logger.info(
                            f"会话 {session_id}: TTS模块成功生成音频，类型: {tts_output_audio_data.type}，长度: {len(tts_output_audio_data.audio)}字节。")
                        tts_audio_b64_output = base64.b64encode(tts_output_audio_data.audio).decode('utf-8')
                    elif tts_output_audio_data:
                        logger.warning(f"会话 {session_id}: TTS模块返回的AudioData对象中没有音频字节。")
                    else:
                        logger.warning(f"会话 {session_id}: TTS模块未生成有效的音频数据。")
            elif not self.tts_module:
                logger.info(f"会话 {session_id}: TTS模块不可用，跳过TTS处理。")

            # 4. 准备最终回复
            return {
                "session_id": session_id,
                "text": final_response_text,
                "audio_b64": tts_audio_b64_output,  # 如果TTS失败或不可用，此值为None
                "error": None  # 表示主要流程成功
            }

        except Exception as e:  # 捕获流程中未被特定捕获的任何其他异常
            logger.exception(f"ChatEngine在处理会话 {session_id} 消息时发生未处理的内部错误: {e}")
            return {
                "session_id": session_id,
                "text": "抱歉，处理您的请求时发生了一个意外的系统错误。",
                "audio_b64": None,
                "error": f"CHAT_ENGINE_INTERNAL_ERROR: {str(e)}"
            }
