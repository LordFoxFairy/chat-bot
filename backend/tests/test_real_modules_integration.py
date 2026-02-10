"""真实模块集成测试 - 模块间互相验证"""
import asyncio
import os
import sys

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils.logging_setup import logger
from backend.utils.config_loader import ConfigLoader
from backend.core.app_context import AppContext
from backend.core.engine.chat_engine import ChatEngine
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.core.models import TextData, AudioData


async def test_tts_module():
    """测试 TTS 模块 - 生成真实音频"""
    logger.info("\n===== 测试 TTS 模块（Edge TTS）=====")

    # 加载测试配置
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'test_config.yaml')
    config = await ConfigLoader.load_config(config_path)

    if not config or "tts" not in config.get("modules", {}):
        logger.error("TTS 配置未找到")
        return False

    try:
        # 创建 ChatEngine
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)
        engine = ChatEngine(config=config, session_manager=session_manager)

        await engine.initialize()
        logger.info("✓ ChatEngine 初始化成功")

        # 获取 TTS 模块
        tts_module = AppContext.get_module("tts")
        if not tts_module:
            logger.error("TTS 模块未初始化")
            return False

        logger.info(f"✓ TTS 模块加载: {tts_module.__class__.__name__}")

        # 测试流式合成
        test_text = "你好，这是一段测试音频。"
        logger.info(f"合成文本: {test_text}")

        text_data = TextData(text=test_text, is_final=True)

        # 收集所有音频块
        logger.info("开始流式合成...")
        chunk_count = 0
        total_bytes = 0
        audio_chunks = []

        async for audio_chunk in tts_module.synthesize_stream(text_data):
            chunk_count += 1
            total_bytes += len(audio_chunk.data)
            audio_chunks.append(audio_chunk.data)

        logger.info(f"✓ 流式合成完成:")
        logger.info(f"  - 音频块数: {chunk_count}")
        logger.info(f"  - 总字节数: {total_bytes}")

        # 合并所有音频块并保存
        if audio_chunks:
            output_dir = os.path.join(PROJECT_ROOT, 'tests', 'test_output')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'tts_test_output.mp3')

            with open(output_file, 'wb') as f:
                for chunk in audio_chunks:
                    f.write(chunk)

            logger.info(f"✓ 音频已保存: {output_file}")

        logger.info("\n✅ TTS 模块测试通过\n")
        return True

    except Exception as e:
        logger.error(f"TTS 测试失败: {e}", exc_info=True)
        return False
    finally:
        AppContext.clear()


async def test_chat_engine_with_real_modules():
    """测试 ChatEngine 加载真实模块"""
    logger.info("\n===== 测试 ChatEngine 加载真实模块 =====")

    config_path = os.path.join(PROJECT_ROOT, 'configs', 'test_config.yaml')
    config = await ConfigLoader.load_config(config_path)

    try:
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)
        engine = ChatEngine(config=config, session_manager=session_manager)

        await engine.initialize()
        logger.info("✓ ChatEngine 初始化成功")

        # 检查加载的模块
        logger.info("\n已加载的模块:")
        for module_name in ["tts", "asr", "llm", "vad"]:
            module = AppContext.get_module(module_name)
            if module:
                logger.info(f"  ✓ {module_name}: {module.__class__.__name__}")
            else:
                logger.info(f"  - {module_name}: 未配置")

        # 验证 TTS 模块可用
        tts = AppContext.get_module("tts")
        if tts:
            assert tts.is_ready, "TTS 模块未就绪"
            logger.info(f"\n✓ TTS 模块就绪: {tts.__class__.__name__}")

        logger.info("\n✅ ChatEngine 真实模块加载测试通过\n")
        return True

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False
    finally:
        AppContext.clear()


async def main():
    """运行所有真实模块集成测试"""
    logger.info("\n" + "="*60)
    logger.info("开始真实模块集成测试")
    logger.info("="*60)

    try:
        # 测试 ChatEngine 加载
        success1 = await test_chat_engine_with_real_modules()

        # 测试 TTS 模块
        success2 = await test_tts_module()

        if success1 and success2:
            logger.info("\n" + "="*60)
            logger.info("✅ 所有真实模块集成测试通过")
            logger.info("="*60 + "\n")
        else:
            logger.error("\n❌ 测试失败")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"\n测试异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
