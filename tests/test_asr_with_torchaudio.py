"""使用torchaudio后端测试ASR"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置环境变量使用torchaudio而不是ffmpeg
os.environ['FUNASR_AUDIO_BACKEND'] = 'torchaudio'

from src.utils.logging_setup import logger
from src.utils.config_loader import ConfigLoader
from src.core.app_context import AppContext
from src.core.engine.chat_engine import ChatEngine
from src.core.session.session_manager import SessionManager, InMemoryStorage
from src.core.models import AudioData
import asyncio


async def test_asr_with_example():
    """使用模型自带示例测试ASR"""
    logger.info("\n===== 测试 ASR 模块（使用示例文件）=====")

    config_path = os.path.join(PROJECT_ROOT, 'configs', 'test_config.yaml')
    config = await ConfigLoader.load_config(config_path)

    try:
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)
        engine = ChatEngine(config=config, session_manager=session_manager)
        await engine.initialize()

        asr_module = AppContext.get_module("asr")
        if not asr_module:
            logger.error("ASR 模块未初始化")
            return False

        logger.info(f"✓ ASR 模块加载: {asr_module.__class__.__name__}")

        # 使用模型自带的中文示例
        example_file = os.path.join(
            PROJECT_ROOT,
            'models/asr/SenseVoiceSmall/iic/SenseVoiceSmall/example/zh.mp3'
        )

        if not os.path.exists(example_file):
            logger.error(f"示例文件不存在: {example_file}")
            return False

        logger.info(f"使用示例文件: {example_file}")

        # 读取音频文件
        with open(example_file, 'rb') as f:
            audio_bytes = f.read()

        logger.info(f"音频大小: {len(audio_bytes)} 字节")

        # 创建AudioData
        audio_data = AudioData(
            data=audio_bytes,
            format="mp3",
            sample_rate=16000,
            channels=1,
        )

        # 识别
        logger.info("开始识别...")
        text = await asr_module.recognize(audio_data)

        logger.info(f"✓ 识别结果: '{text}'")

        if text and len(text.strip()) > 0:
            logger.info("✅ ASR 测试成功")
            return True
        else:
            logger.warning("ASR 未识别到文本")
            return False

    except Exception as e:
        logger.error(f"ASR 测试失败: {e}", exc_info=True)
        return False
    finally:
        AppContext.clear()


if __name__ == "__main__":
    asyncio.run(test_asr_with_example())
