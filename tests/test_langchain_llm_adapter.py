import asyncio
import logging
import os
import sys
import time
from typing import List, Optional, Dict, Any


# --- 基礎日誌配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 核心框架组件导入 ---
from data_models.text_data import TextData
from services.config_loader import ConfigLoader
from adapters.llm.langchain_llm_adapter import LangchainLLMAdapter
from core_framework.session_manager import SessionManager
from core_framework.exceptions import ModuleInitializationError, ModuleProcessingError


# EventManager 可以為 None，如果適配器不直接依賴它發送特定事件
# from core_framework.event_manager import EventManager


async def run_langchain_llm_adapter_test():
    logger.info("LangchainLLMAdapter 直接實例化測試: 開始...")
    logger.info(f"Python sys.path: {sys.path}")
    logger.info(f"當前工作目錄: {os.getcwd()}")

    # 相對於 CHAT_BOT_ROOT 的配置文件路徑
    config_file_path = os.path.join(
        os.path.dirname(__file__),
        "../configs",
        "config.yaml"  # Ensure this file exists and is configured for "fake" provider
    )
    logger.info(f"嘗試加載配置文件: {config_file_path}")

    if not os.path.exists(config_file_path):
        logger.error(f"配置文件未找到: {config_file_path}")
        return

    try:
        config_data = ConfigLoader.load_config(config_file_path)
        if not config_data:
            logger.error(f"測試失敗: 配置文件 '{config_file_path}' 加載失敗或為空。")
            return
    except Exception as e:
        logger.error(f"測試失敗: 配置加載錯誤 ('{config_file_path}'): {e}", exc_info=True)
        return

    event_loop = asyncio.get_event_loop()
    session_manager = SessionManager(default_max_history_turns=5)  # 實例化 SessionManager
    event_manager = None  # 對於此適配器單元測試，可以為 None

    # 從配置文件中提取 LLM 模塊的頂層配置
    llm_module_overall_config: Optional[Dict[str, Any]] = config_data.get("modules", {}).get("llm")
    if not llm_module_overall_config:
        logger.error("測試失敗: 配置文件中未找到 'modules.llm' 部分。")
        return

    # 檢查 API Key 配置 (以 DeepSeek 為例，根據 config.yaml)
    # 實際測試時，確保環境變量 DEEPSEEK_API_KEY 已設置，或者 config.yaml 中直接提供了有效的 key
    # 這裡的 config.yaml 似乎直接包含了 api_key
    enabled_provider = llm_module_overall_config.get("enable_module")
    provider_configs = llm_module_overall_config.get("config", {}).get(enabled_provider, {})
    api_key = provider_configs.get("api_key")
    api_key_env = provider_configs.get("api_key_env_var")

    if not api_key and api_key_env and not os.getenv(api_key_env):
        logger.warning(f"警告: LLM 提供商 '{enabled_provider}' 可能需要 API 密鑰。")
        logger.warning(f"配置文件指向環境變量 '{api_key_env}'，但該變量未設置。")
        logger.warning("如果測試失敗，請檢查 API 密鑰配置。")
    elif not api_key and not api_key_env:
        logger.warning(f"警告: LLM 提供商 '{enabled_provider}' 未配置 API 密鑰或環境變量。")

    test_module_id = "langchain_llm_direct_test_instance"
    loaded_llm_adapter: Optional[LangchainLLMAdapter] = None

    try:
        logger.info(f"測試: 直接實例化 LangchainLLMAdapter (ID: {test_module_id})")
        logger.debug(f"  傳遞給適配器的頂層 LLM 配置: {llm_module_overall_config}")

        loaded_llm_adapter = LangchainLLMAdapter(
            module_id=test_module_id,
            config=llm_module_overall_config,
            event_loop=event_loop,
            event_manager=event_manager,
            session_manager=session_manager
        )

        logger.info(f"測試: 調用 {test_module_id}.initialize()...")
        await loaded_llm_adapter.initialize()

        if not loaded_llm_adapter.is_ready:
            logger.error(f"測試失敗: 適配器 {test_module_id} 初始化後未就緒。")
            return

    except ModuleInitializationError as e_init_mod:
        logger.error(f"測試失敗: 適配器初始化期間發生 ModuleInitializationError: {e_init_mod}", exc_info=True)
        return
    except Exception as e_init:
        logger.error(f"測試失敗: 適配器實例化或初始化期間发生未知错误: {e_init}", exc_info=True)
        return

    logger.info(f"測試: 適配器 '{test_module_id}' (類型: {type(loaded_llm_adapter).__name__}) 創建並初始化成功。")
    logger.info(f"  - LLM 提供商 (來自配置): {loaded_llm_adapter.enabled_provider_name}")
    logger.info(f"  - 模型名稱 (來自配置): {loaded_llm_adapter.model_name}")

    session_id = f"test_direct_llm_sess_{int(time.time())}"
    session_manager.create_session(session_id)  # 確保會話存在
    logger.info(f"測試會話 ID: {session_id}")

    # --- 1. 測試 generate_complete_response (非流式) ---
    logger.info(f"\n--- [測試 1/2] 開始測試 generate_complete_response (非流式) ---")
    try:
        non_stream_input_text = "你好，请介绍一下大型语言模型。"
        non_stream_input_data = TextData(text=non_stream_input_text, chunk_id=f"{session_id}_ns_input")

        logger.info(f"調用 generate_complete_response，輸入: '{non_stream_input_text}'")
        # BaseLLM.generate_complete_response 内部会处理历史记录的添加
        non_stream_response_data = await loaded_llm_adapter.generate_complete_response(non_stream_input_data,
                                                                                       session_id)

        if non_stream_response_data and isinstance(non_stream_response_data, TextData):
            logger.info(
                f"\033[92m[generate_complete_response 結果]: '{non_stream_response_data.text[:200]}...' (is_final={non_stream_response_data.is_final})\033[0m")
            if not non_stream_response_data.is_final:
                logger.warning("[generate_complete_response 警告]: 返回的 TextData is_final 應為 True。")
            if not non_stream_response_data.text.strip():
                logger.warning("[generate_complete_response 警告]: LLM 返回了空文本。")
        elif non_stream_response_data is None:
            logger.error(
                "[generate_complete_response 錯誤]: 方法返回了 None，預期 TextData。檢查日誌中的 _handle_llm_error。")
        else:
            logger.error(f"[generate_complete_response 錯誤]: 返回了非預期類型: {type(non_stream_response_data)}。")

        # 檢查歷史記錄
        history = session_manager.get_dialogue_history(session_id)
        logger.info(
            f"非流式測試後，會話 '{session_id}' 的歷史記錄 (最近2條): {history[-2:] if len(history) >= 2 else history}")
        if not any(entry['role'] == 'user' and entry['content'] == non_stream_input_text for entry in history):
            logger.warning(f"歷史記錄中未找到用戶输入: '{non_stream_input_text}'")
        if non_stream_response_data and non_stream_response_data.text and \
                not any(entry['role'] == 'assistant' and entry['content'] == non_stream_response_data.text for entry in
                        history):
            logger.warning(f"歷史記錄中未找到助手响应: '{non_stream_response_data.text[:50]}...'")


    except ModuleProcessingError as e_proc:
        logger.error(f"[generate_complete_response 測試] 發生 ModuleProcessingError: {e_proc}", exc_info=True)
    except Exception as e_block:
        logger.error(f"[generate_complete_response 測試] 發生未知错误: {e_block}", exc_info=True)
    logger.info(f"--- [測試 1/2] 結束測試 generate_complete_response ---\n")

    # 清理一下历史，避免影响流式测试的上下文（或者使用新的session_id）
    # 为了简单，这里继续用同一个session_id，但注意上下文会累积
    # session_manager.end_session(session_id)
    # session_id = f"test_direct_llm_sess_stream_{int(time.time())}"
    # session_manager.create_session(session_id)
    # logger.info(f"为流式测试重置会话 ID: {session_id}")

    # --- 2. 測試 stream_chat_response (流式) ---
    logger.info(f"--- [測試 2/2] 開始測試 stream_chat_response (流式) ---")
    stream_results_text: List[str] = []
    final_streamed_data: Optional[TextData] = None
    try:
        stream_input_text = "用三个词总结一下今天的天气怎么样？"
        stream_input_data = TextData(text=stream_input_text, chunk_id=f"{session_id}_s_input")

        logger.info(f"調用 stream_chat_response，輸入: '{stream_input_text}'")
        # BaseLLM.stream_chat_response 内部会处理历史记录的添加和最终事件
        async for text_data_chunk in loaded_llm_adapter.stream_chat_response(stream_input_data, session_id):
            if text_data_chunk and isinstance(text_data_chunk, TextData):
                log_level_stream = logging.INFO if text_data_chunk.is_final else logging.DEBUG
                logger.log(log_level_stream,
                           f"[stream_chat_response 块]: '{text_data_chunk.text}' (is_final={text_data_chunk.is_final}, meta={text_data_chunk.metadata})")
                if text_data_chunk.text:  # 只收集非空文本块
                    stream_results_text.append(text_data_chunk.text)
                if text_data_chunk.is_final:
                    final_streamed_data = text_data_chunk
                    # is_final=True的块可能是包含累积文本的最终块，也可能是一个空文本的错误指示块
                    if "error" in text_data_chunk.metadata:
                        logger.error(f"流式响应以错误结束: {text_data_chunk.metadata['error']}")
            else:  # pragma: no cover (适配器应总是yield TextData)
                logger.warning(f"[stream_chat_response 块]: 收到非TextData或None类型的块: {text_data_chunk}")

        logger.info(f"--- stream_chat_response 流處理完畢 ---")

        if not stream_results_text:
            logger.warning("[stream_chat_response 測試]: 未收到任何流式文本块。")

        if final_streamed_data:
            logger.info(
                f"\033[92m[stream_chat_response 最终数据]: Text='{final_streamed_data.text}', Final={final_streamed_data.is_final}, Meta={final_streamed_data.metadata}\033[0m")
            if not final_streamed_data.is_final:
                logger.warning("[stream_chat_response 測試]: 标记为最终的流数据块的 is_final 标志不是 True。")

            full_streamed_response = "".join(stream_results_text)
            # The final_streamed_data.text from BaseLLM's stream_chat_response (after _handle_llm_success)
            # should be the complete accumulated text.
            if full_streamed_response.strip() and final_streamed_data.text.strip() != full_streamed_response.strip():
                logger.warning(
                    f"[stream_chat_response 測試]: 拼接的流文本 ('{full_streamed_response[:50]}...') 与最终数据中的文本 ('{final_streamed_data.text[:50]}...') 不完全一致。这可能是正常的，如果最终块是特殊标记。")
            elif not full_streamed_response.strip() and final_streamed_data.text.strip():
                logger.info("[stream_chat_response 測試]: 流式块累积文本为空，但最终数据块包含文本。")
            elif full_streamed_response.strip() and not final_streamed_data.text.strip() and "error" not in final_streamed_data.metadata:
                logger.warning("[stream_chat_response 測試]: 流式块累积文本非空，但最终数据块文本为空且无错误。")


        else:
            logger.warning("[stream_chat_response 測試]: 未收到标记为 is_final=True 的最终流数据块。")

        # 檢查歷史記錄
        history_stream = session_manager.get_dialogue_history(session_id)
        logger.info(
            f"流式測試後，會話 '{session_id}' 的歷史記錄 (最近2條): {history_stream[-2:] if len(history_stream) >= 2 else history_stream}")
        if not any(entry['role'] == 'user' and entry['content'] == stream_input_text for entry in history_stream):
            logger.warning(f"歷史記錄中未找到用戶输入: '{stream_input_text}'")

        combined_stream_text = "".join(stream_results_text).strip()
        if combined_stream_text and \
                not any(entry['role'] == 'assistant' and entry['content'].strip() == combined_stream_text for entry in
                        history_stream):
            # 历史记录中的助手回复应该是完整的，而不是分块的
            # 检查是否有包含这个组合文本的助手回复
            found_assistant_response = False
            for entry in history_stream:
                if entry['role'] == 'assistant' and combined_stream_text in entry['content'].strip():
                    found_assistant_response = True
                    break
            if not found_assistant_response:
                logger.warning(f"歷史記錄中未找到与流式响应 '{combined_stream_text[:50]}...' 完全匹配的助手响应。")


    except ModuleProcessingError as e_proc_stream:
        logger.error(f"[stream_chat_response 測試] 發生 ModuleProcessingError: {e_proc_stream}", exc_info=True)
    except Exception as e_stream:
        logger.error(f"[stream_chat_response 測試] 發生未知错误: {e_stream}", exc_info=True)

    logger.info(f"\nLangchainLLMAdapter 直接測試: 收集到的所有流式文本片段: {len(stream_results_text)} 条")
    logger.info(f"  拼接后的流式响应: '{''.join(stream_results_text)[:200]}...'")

    await asyncio.sleep(0.1)  # 短暂等待，确保所有异步任务完成
    logger.info(f"LangchainLLMAdapter [{test_module_id}] 測試: 准备调用 dispose()...")
    if loaded_llm_adapter and hasattr(loaded_llm_adapter, 'dispose') and callable(loaded_llm_adapter.dispose):
        await loaded_llm_adapter.dispose()
    logger.info(f"LangchainLLMAdapter [{test_module_id}] 測試: dispose() 调用完成。")
    logger.info("LangchainLLMAdapter 直接測試: 測試結束。")


if __name__ == "__main__":
    # 确保项目根目录在 sys.path 中，以便导入模块
    # 此处的 CHAT_BOT_ROOT 应该指向包含 data_models, services, modules, adapters 的目录
    # 例如，如果 tests 目录与 chat-bot 目录同级，则 CHAT_BOT_ROOT 应该是 'chat-bot'
    # 如果 tests 在 chat-bot 内部，则 CHAT_BOT_ROOT 应该是 chat-bot 的父目录的 chat-bot 子目录，即 chat-bot 本身

    # 修正路径，假设此脚本位于 chat-bot/tests/adapters/llm/
    # 则 CHAT_BOT_ROOT 应该是 chat-bot/
    # current_file_dir = os.path.dirname(os.path.abspath(__file__)) # .../chat-bot/tests/adapters/llm
    # adapters_dir = os.path.dirname(current_file_dir) # .../chat-bot/tests/adapters
    # tests_dir = os.path.dirname(adapters_dir) # .../chat-bot/tests
    # project_root_for_imports = os.path.dirname(tests_dir) # .../chat-bot/
    # if project_root_for_imports not in sys.path:
    #     sys.path.insert(0, project_root_for_imports)
    # logger.info(f"将项目根目录 '{project_root_for_imports}' 添加到 sys.path 以进行模块导入。")
    # (上面的路径设置已在文件顶部处理)

    try:
        asyncio.run(run_langchain_llm_adapter_test())
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("\n測試被用戶中斷。")
    except Exception as e_main:  # pragma: no cover
        logger.critical(f"\n測試主程序遭遇无法处理的致命错误: {e_main}", exc_info=True)
    finally:
        logger.info("測試程序退出。")

