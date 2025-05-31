# test_single_tts.py
import asyncio
import logging
import os
import sys
import uuid

import yaml  # 導入 PyYAML 庫
from typing import Dict, Any, Optional

# 假設導入路徑已正確設置
from data_models.text_data import TextData
# from data_models.audio_data import AudioData # 主要用於類型提示，測試依賴 BaseTTS 保存
from adapters.tts.edge_tts_adapter import EdgeTTSAdapter
from core_framework.exceptions import ModuleInitializationError  # 假設的異常
from services.config_loader import ConfigLoader

# 配置日誌記錄
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    從指定的 YAML 文件路徑加載配置。
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置已成功從 {config_path} 加載。")
        return config
    except FileNotFoundError:
        logger.error(f"配置文件 {config_path} 未找到。")
        return None
    except yaml.YAMLError as e:
        logger.error(f"解析 YAML 配置文件 {config_path} 失敗: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"加載配置文件 {config_path} 時發生未知錯誤: {e}", exc_info=True)
        return None


async def main():
    """
    主測試函數。
    """
    # 1. 定義 YAML 配置文件路徑
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


    # 從配置文件中提取 ASR 模塊的頂層配置
    # (即 YAML 中 modules.asr 下的整個字典)
    config = config_data.get("modules", {}).get("tts")

    # 3. 實例化 EdgeTTSAdapter
    try:
        tts_adapter = EdgeTTSAdapter(
            config=config
        )
        logger.info(f"EdgeTTSAdapter 實例化成功 (ID: {tts_adapter.module_id})")
    except Exception as e:
        logger.error(f"實例化 EdgeTTSAdapter 失敗: {e}", exc_info=True)
        return

    # 4. 初始化適配器
    await tts_adapter.initialize()


    # 5. 準備輸入文本
    text_to_synthesize = "你好，世界！這是根據用戶提供的 YAML 結構進行的 Edge TTS 適配器測試。"
    input_text_data = TextData(
        text=text_to_synthesize,
        chunk_id=f"user_yaml_tts_stream_{uuid.uuid4().hex[:6]}",  # 動態生成 chunk_id
        lang="zh-CN"
    )
    logger.info(f"準備合成文本: '{text_to_synthesize}' (Stream ID: {input_text_data.chunk_id})")

    # 6. 運行 TTS 並處理音頻流
    all_received_audio_bytes = bytearray()
    first_audio_chunk_details: Optional[Dict[str, Any]] = None
    chunk_count = 0

    logger.info("開始調用 TTS 適配器的 run 方法...")
    try:
        async for audio_chunk in tts_adapter.run(input_text_data):
            chunk_count += 1
            logger.info(f"  收到音頻塊 {chunk_count}: "
                        f"格式={audio_chunk.format.value}, "
                        f"採樣率={audio_chunk.sample_rate}, "
                        f"聲道={audio_chunk.channels}, "
                        f"採樣寬度={audio_chunk.sample_width}, "
                        f"數據長度={len(audio_chunk.data)} 字節, "
                        f"是否最終塊={audio_chunk.is_final}")

            if audio_chunk.data:
                all_received_audio_bytes.extend(audio_chunk.data)
                if first_audio_chunk_details is None:
                    first_audio_chunk_details = {
                        "format": audio_chunk.format.value,
                        "sample_rate": audio_chunk.sample_rate,
                        "channels": audio_chunk.channels,
                        "sample_width": audio_chunk.sample_width,
                    }

            if audio_chunk.is_final and not audio_chunk.data:
                logger.info("  收到最終的空音頻塊，表示流結束。")



    except Exception as e:
        logger.error(f"運行 TTS 或處理音頻流時發生錯誤: {e}", exc_info=True)

    logger.info("TTS 測試完成。")


if __name__ == "__main__":
    if sys.version_info >= (3, 7):
        asyncio.run(main())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())