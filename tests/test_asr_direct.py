"""直接使用FunASR测试ASR（跳过audio_converter）"""
import os
import sys
import asyncio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logging_setup import logger
from src.utils.config_loader import ConfigLoader
from src.core.app_context import AppContext
from src.core.engine.chat_engine import ChatEngine
from src.core.session.session_manager import SessionManager, InMemoryStorage


async def test_asr_direct():
    """直接使用FunASR模型测试"""
    logger.info("\n===== 直接测试 ASR 模块 =====")

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

        # 直接使用FunASR模型的generate方法
        example_file = os.path.join(
            PROJECT_ROOT,
            'models/asr/SenseVoiceSmall/iic/SenseVoiceSmall/example/zh.mp3'
        )

        logger.info(f"使用示例文件: {example_file}")

        # 直接调用模型
        if asr_module.model:
            logger.info("开始识别（直接调用FunASR）...")

            result = asr_module.model.generate(
                input=example_file,
            )

            logger.info(f"识别结果: {result}")

            if result and isinstance(result, list) and len(result) > 0:
                text = result[0].get("text", "")
                logger.info(f"\n✅ ASR 测试成功")
                logger.info(f"识别文本: '{text}'")
                return True
            else:
                logger.warning("未识别到文本")
                return False
        else:
            logger.error("ASR 模型未加载")
            return False

    except Exception as e:
        logger.error(f"ASR 测试失败: {e}", exc_info=True)
        return False
    finally:
        AppContext.clear()


if __name__ == "__main__":
    asyncio.run(test_asr_direct())
