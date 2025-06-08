import asyncio
from utils.logging_setup import logger
import os
import sys
import time
from typing import AsyncGenerator, Optional, List  # List 用於收集結果

# --- 基礎日誌配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- 路徑設置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
logger.info(f"項目根目錄 '{PROJECT_ROOT}' 已添加到sys.path。")

# --- 核心框架组件导入 ---
from data_models import AudioData, TextData, AudioFormat
from utils.config_loader import ConfigLoader
from modules.base_asr import BaseASR  # 用於類型提示
# 直接導入要測試的適配器
from adapters.asr.funasr_sensevoice_adapter import FunASRSenseVoiceAdapter


async def simulate_audio_bytes_stream(duration_s: float = 2.0, sample_rate: int = 16000, channels: int = 1,
                                      sample_width: int = 2, chunk_duration_ms: int = 100) -> AsyncGenerator[
    bytes, None]:
    """模擬音頻輸入流，產生 PCM 字節塊。"""
    bytes_per_sample = channels * sample_width
    samples_per_chunk = int((chunk_duration_ms / 1000) * sample_rate)
    bytes_per_chunk = samples_per_chunk * bytes_per_sample

    num_chunks = int(duration_s * 1000 / chunk_duration_ms)
    if num_chunks == 0 and duration_s > 0:
        num_chunks = 1
        bytes_per_chunk = int(duration_s * sample_rate * channels * sample_width)

    logger.debug(
        f"[模擬音頻字節流] 準備發送 {num_chunks} 個音頻塊, 每個塊約 {chunk_duration_ms}ms, 總時長 approx {duration_s}s")
    for i in range(num_chunks):
        yield b'\x00' * bytes_per_chunk
        await asyncio.sleep(chunk_duration_ms / 1000)
    logger.debug(f"[模擬音頻字節流] 所有音頻塊發送完毕。")


async def create_audio_data_stream(
        asr_module: BaseASR,  # 仍然使用 BaseASR 類型提示以獲取期望的音頻參數
        duration_s: float = 2.5,
        chunk_duration_ms: int = 200
) -> AsyncGenerator[AudioData, None]:
    """
    創建一個 AudioData 對象的異步生成器，用於直接饋送給 ASR 模塊的 stream_recognize_audio。
    """
    logger.info(f"[AudioData 生成器] 開始生成 AudioData 流 (時長: {duration_s}s, 塊間隔: {chunk_duration_ms}ms)...")
    async for pcm_chunk_bytes in simulate_audio_bytes_stream(
            duration_s=duration_s,
            sample_rate=asr_module.expected_sample_rate,
            channels=asr_module.expected_channels,
            sample_width=asr_module.expected_sample_width,
            chunk_duration_ms=chunk_duration_ms
    ):
        yield AudioData(
            data=pcm_chunk_bytes,
            format=AudioFormat.PCM,
            sample_rate=asr_module.expected_sample_rate,
            channels=asr_module.expected_channels,
            sample_width=asr_module.expected_sample_width
        )
    logger.info(f"[AudioData 生成器] AudioData 流生成完畢。")


async def run_funasr_adapter_direct_test():
    logger.info("FunASRSenseVoiceAdapter 直接實例化測試: 開始...")

    logger.info(f"當前工作目錄: {os.getcwd()}")

    config_file_path = os.path.join(
        os.path.dirname(__file__),
        "../configs",
        "config.yaml"  # Ensure this file exists and is configured for "fake" provider
    )
    config_file_path = os.path.abspath(config_file_path)
    logger.info(f"嘗試加載配置文件: {config_file_path}")

    try:
        config_data = ConfigLoader.load_config(config_file_path)
        if not config_data:
            logger.error(f"測試失敗: 配置文件 '{config_file_path}' 加載失敗或為空。")
            return
    except Exception as e:
        logger.error(f"測試失敗: 配置加載錯誤 ('{config_file_path}'): {e}", exc_info=True)
        return

    event_loop = asyncio.get_event_loop()

    # 從配置文件中提取 ASR 模塊的頂層配置
    # (即 YAML 中 modules.asr 下的整個字典)
    asr_module_overall_config = config_data.get("modules", {}).get("asr")
    if not asr_module_overall_config:
        logger.error("測試失敗: 配置文件中未找到 'modules.asr' 部分。")
        return

    # 直接實例化 FunASRSenseVoiceAdapter
    # module_id 可以是一個測試特定的ID
    test_module_id = "funasr_direct_test_instance"
    loaded_asr_module: Optional[BaseASR] = None  # 類型提示為 BaseASR

    try:
        logger.info(f"測試: 直接實例化 FunASRSenseVoiceAdapter (ID: {test_module_id})")
        logger.debug(f"  傳遞給適配器的頂層 ASR 配置: {asr_module_overall_config}")

        loaded_asr_module = FunASRSenseVoiceAdapter(
            module_id=test_module_id,
            config=asr_module_overall_config,  # 傳遞 modules.asr 下的整個配置字典
            event_loop=event_loop,
            event_manager=None  # 如果適配器內部不直接使用 event_manager，可以傳 None
        )

        logger.info(f"測試: 調用 {test_module_id}.initialize()...")
        await loaded_asr_module.initialize()  # 手動調用初始化

        if not loaded_asr_module.is_ready:
            logger.error(f"測試失敗: 適配器 {test_module_id} 初始化後未就緒。")
            return

    except Exception as e_init:
        logger.error(f"測試失敗: 適配器實例化或初始化期間發生錯誤: {e_init}", exc_info=True)
        return

    logger.info(f"測試: 適配器 '{test_module_id}' (類型: {type(loaded_asr_module).__name__}) 創建並初始化成功。")
    logger.info(f"  - ASR 引擎名稱 (來自配置): {loaded_asr_module.asr_engine_name}")
    logger.info(f"  - 期望採樣率: {loaded_asr_module.expected_sample_rate} Hz")

    session_id = f"test_direct_asr_sess_{int(time.time())}"
    logger.info(f"測試會話 ID: {session_id}")

    # --- 1. 測試 recognize_audio_block ---
    logger.info(f"\n--- [測試 1/2] 開始測試 recognize_audio_block ---")
    try:
        single_chunk_bytes = b''
        async for chunk in simulate_audio_bytes_stream(
                duration_s=1.0,
                sample_rate=loaded_asr_module.expected_sample_rate,
                channels=loaded_asr_module.expected_channels,
                sample_width=loaded_asr_module.expected_sample_width,
                chunk_duration_ms=1000
        ):
            single_chunk_bytes = chunk
            break

        if single_chunk_bytes:
            block_audio_data = AudioData(
                data=single_chunk_bytes,
                format=AudioFormat.PCM,
                sample_rate=loaded_asr_module.expected_sample_rate,
                channels=loaded_asr_module.expected_channels,
                sample_width=loaded_asr_module.expected_sample_width
            )
            logger.info(f"調用 recognize_audio_block，音頻數據長度: {len(block_audio_data.data)} 字節")
            block_text_data = await loaded_asr_module.recognize_audio_block(block_audio_data, session_id)

            if block_text_data:
                logger.info(
                    f"\033[92m[recognize_audio_block 結果]: '{block_text_data.text}' (is_final={block_text_data.is_final})\033[0m")
                if not block_text_data.is_final:
                    logger.warning("[recognize_audio_block 警告]: 返回的 TextData is_final 應為 True。")
            else:  # BaseASR.recognize_audio_block 在 is_final=True 時應總返回 TextData
                logger.error("[recognize_audio_block 錯誤]: 未返回預期的 TextData 對象。")
        else:
            logger.warning("[recognize_audio_block 測試]: 未能生成測試音頻塊。")
    except Exception as e_block:
        logger.error(f"[recognize_audio_block 測試] 發生錯誤: {e_block}", exc_info=True)
    logger.info(f"--- [測試 1/2] 結束測試 recognize_audio_block ---\n")

    # --- 2. 測試 stream_recognize_audio ---
    logger.info(f"--- [測試 2/2] 開始測試 stream_recognize_audio ---")
    stream_results: List[TextData] = []
    try:
        audio_stream_for_asr = create_audio_data_stream(
            asr_module=loaded_asr_module,
            duration_s=3.5,
            chunk_duration_ms=300
        )

        logger.info(f"調用 stream_recognize_audio...")
        async for text_data_result in loaded_asr_module.stream_recognize_audio(audio_stream_for_asr, session_id):
            if text_data_result:
                stream_results.append(text_data_result)
                log_level = logging.INFO if text_data_result.is_final else logging.DEBUG
                logger.log(log_level,
                           f"[stream_recognize_audio 結果]: '{text_data_result.text}' (is_final={text_data_result.is_final})"
                           )
            else:  # BaseASR.stream_recognize_audio 在 is_final=False 且文本為空時可能返回 None
                logger.debug("[stream_recognize_audio 結果]: 收到 None (可能是空的中间结果)。")

        logger.info(f"--- stream_recognize_audio 流處理完畢 ---")
        if not stream_results:
            logger.warning("[stream_recognize_audio 測試]: 未收到任何流式結果。")
        elif not stream_results[-1].is_final:
            logger.warning("[stream_recognize_audio 測試]: 最後一個流式結果的 is_final 標誌不是 True。")

    except Exception as e_stream:
        logger.error(f"[stream_recognize_audio 測試] 發生錯誤: {e_stream}", exc_info=True)

    logger.info(f"\nFunASRSenseVoiceAdapter 直接測試: 收集到的所有流式 ASR 結果: {len(stream_results)} 條")
    for i, td in enumerate(stream_results):
        logger.info(f"  結果 {i + 1}: Text: '{td.text}', Final: {td.is_final}")

    await asyncio.sleep(0.1)
    logger.info("FunASRSenseVoiceAdapter 直接測試: 準備關閉適配器 (如果需要)...")
    if hasattr(loaded_asr_module, 'stop') and callable(loaded_asr_module.stop):
        await loaded_asr_module.stop()  # 如果 BaseModule 有 stop 方法
    logger.info("FunASRSenseVoiceAdapter 直接測試: 測試結束。")


if __name__ == "__main__":
    try:
        asyncio.run(run_funasr_adapter_direct_test())
    except KeyboardInterrupt:
        logger.info("\n測試被用戶中斷。")
    except Exception as e_main:
        logger.error(f"\n測試主程序異常: {e_main}", exc_info=True)
    finally:
        logger.info("測試程序退出。")

