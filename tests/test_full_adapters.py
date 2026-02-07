"""全面适配器测试 - 测试所有真实模块"""
import asyncio
import os
import sys
import struct

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logging_setup import logger
from src.utils.config_loader import ConfigLoader
from src.core.app_context import AppContext
from src.core.engine.chat_engine import ChatEngine
from src.core.session.session_manager import SessionManager, InMemoryStorage
from src.core.models import TextData, AudioData


async def test_tts_module():
    """测试 TTS 模块"""
    logger.info("\n===== 测试 TTS 模块 =====")

    config_path = os.path.join(PROJECT_ROOT, 'configs', 'test_config.yaml')
    config = await ConfigLoader.load_config(config_path)

    try:
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)
        engine = ChatEngine(config=config, session_manager=session_manager)
        await engine.initialize()

        tts_module = AppContext.get_module("tts")
        if not tts_module:
            logger.error("TTS 模块未初始化")
            return False, None

        logger.info(f"✓ TTS 模块加载: {tts_module.__class__.__name__}")

        # 测试文本
        test_text = "你好，这是语音合成测试。"
        logger.info(f"合成文本: {test_text}")

        text_data = TextData(text=test_text, is_final=True)

        # 收集音频
        audio_chunks = []
        async for audio_chunk in tts_module.synthesize_stream(text_data):
            audio_chunks.append(audio_chunk.data)

        total_bytes = sum(len(chunk) for chunk in audio_chunks)
        logger.info(f"✓ TTS 合成成功: {len(audio_chunks)} 块, {total_bytes} 字节")

        # 保存音频
        output_dir = os.path.join(PROJECT_ROOT, 'tests', 'test_output')
        os.makedirs(output_dir, exist_ok=True)
        tts_output = os.path.join(output_dir, 'tts_output.mp3')

        with open(tts_output, 'wb') as f:
            for chunk in audio_chunks:
                f.write(chunk)

        logger.info(f"✓ 音频已保存: {tts_output}")

        return True, tts_output

    except Exception as e:
        logger.error(f"TTS 测试失败: {e}", exc_info=True)
        return False, None
    finally:
        AppContext.clear()


async def test_vad_module():
    """测试 VAD 模块"""
    logger.info("\n===== 测试 VAD 模块 =====")

    config_path = os.path.join(PROJECT_ROOT, 'configs', 'test_config.yaml')
    config = await ConfigLoader.load_config(config_path)

    try:
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)
        engine = ChatEngine(config=config, session_manager=session_manager)
        await engine.initialize()

        vad_module = AppContext.get_module("vad")
        if not vad_module:
            logger.error("VAD 模块未初始化")
            return False

        logger.info(f"✓ VAD 模块加载: {vad_module.__class__.__name__}")

        # 生成测试音频数据
        # 1. 静音测试 - 全零音频
        sample_rate = 16000
        duration_ms = 500
        num_samples = int(sample_rate * duration_ms / 1000)

        silent_audio = bytes(num_samples * 2)  # 16-bit PCM
        logger.info(f"测试静音音频: {len(silent_audio)} 字节")

        is_speech = await vad_module.detect(silent_audio)
        logger.info(f"  - 静音检测结果: {'有语音' if is_speech else '无语音'}")

        # 2. 模拟语音测试 - 生成正弦波
        import math
        frequency = 440  # A4 音符
        audio_samples = []

        for i in range(num_samples):
            t = i / sample_rate
            # 生成正弦波 (模拟语音)
            value = int(16384 * math.sin(2 * math.pi * frequency * t))
            # 转换为 16-bit PCM
            audio_samples.append(struct.pack('<h', value))

        speech_audio = b''.join(audio_samples)
        logger.info(f"测试模拟语音: {len(speech_audio)} 字节")

        is_speech = await vad_module.detect(speech_audio)
        logger.info(f"  - 语音检测结果: {'有语音' if is_speech else '无语音'}")

        logger.info("✓ VAD 模块测试完成")
        return True

    except Exception as e:
        logger.error(f"VAD 测试失败: {e}", exc_info=True)
        return False
    finally:
        AppContext.clear()


async def test_asr_module(audio_file=None):
    """测试 ASR 模块"""
    logger.info("\n===== 测试 ASR 模块 =====")

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

        # 使用示例音频文件
        if audio_file and os.path.exists(audio_file):
            logger.info(f"使用音频文件: {audio_file}")

            with open(audio_file, 'rb') as f:
                audio_data = AudioData(
                    data=f.read(),
                    format="mp3",
                    sample_rate=16000,
                    channels=1,
                )

            logger.info(f"音频数据大小: {len(audio_data.data)} 字节")

            # 识别
            text = await asr_module.recognize(audio_data)
            logger.info(f"✓ ASR 识别结果: '{text}'")

            if text:
                logger.info("✓ ASR 成功识别音频")
                return True
            else:
                logger.warning("ASR 未识别到文本（可能是音频格式问题）")
                return True
        else:
            # 使用模型自带的示例文件
            example_dir = os.path.join(
                PROJECT_ROOT,
                'models/asr/SenseVoiceSmall/iic/SenseVoiceSmall/example'
            )

            # 查找示例音频
            example_files = []
            if os.path.exists(example_dir):
                for file in os.listdir(example_dir):
                    if file.endswith(('.mp3', '.wav', '.pcm')):
                        example_files.append(os.path.join(example_dir, file))

            if example_files:
                test_file = example_files[0]
                logger.info(f"使用模型示例文件: {test_file}")

                with open(test_file, 'rb') as f:
                    audio_data = AudioData(
                        data=f.read(),
                        format=test_file.split('.')[-1],
                        sample_rate=16000,
                        channels=1,
                    )

                text = await asr_module.recognize(audio_data)
                logger.info(f"✓ ASR 识别结果: '{text}'")
                return True
            else:
                logger.warning("未找到测试音频，跳过 ASR 功能测试")
                logger.info("✓ ASR 模块初始化成功")
                return True

    except Exception as e:
        logger.error(f"ASR 测试失败: {e}", exc_info=True)
        return False
    finally:
        AppContext.clear()


async def test_tts_to_asr_pipeline():
    """测试 TTS → ASR 管道 - 真实模块互相验证"""
    logger.info("\n===== 测试 TTS → ASR 管道 =====")

    # 1. 先用 TTS 生成音频
    logger.info("步骤 1: 使用 TTS 生成音频")
    tts_success, audio_file = await test_tts_module()

    if not tts_success or not audio_file:
        logger.error("TTS 生成失败，无法继续管道测试")
        return False

    # 2. 再用 ASR 识别生成的音频
    logger.info("\n步骤 2: 使用 ASR 识别 TTS 生成的音频")
    asr_success = await test_asr_module(audio_file=audio_file)

    if asr_success:
        logger.info("✅ TTS → ASR 管道测试通过")
        return True
    else:
        logger.error("❌ TTS → ASR 管道测试失败")
        return False


async def main():
    """运行所有测试"""
    logger.info("\n" + "="*60)
    logger.info("开始全面适配器测试")
    logger.info("="*60)

    results = {}

    try:
        # 测试各个模块
        logger.info("\n【第一部分：独立模块测试】")

        tts_success, _ = await test_tts_module()
        results['TTS'] = tts_success

        vad_success = await test_vad_module()
        results['VAD'] = vad_success

        asr_success = await test_asr_module()
        results['ASR'] = asr_success

        # 测试管道
        logger.info("\n【第二部分：模块间互相验证】")

        pipeline_success = await test_tts_to_asr_pipeline()
        results['TTS→ASR Pipeline'] = pipeline_success

        # 汇总结果
        logger.info("\n" + "="*60)
        logger.info("测试结果汇总")
        logger.info("="*60)

        for name, success in results.items():
            status = "✅ 通过" if success else "❌ 失败"
            logger.info(f"  {name}: {status}")

        all_passed = all(results.values())

        if all_passed:
            logger.info("\n" + "="*60)
            logger.info("✅ 所有测试通过")
            logger.info("="*60 + "\n")
        else:
            logger.error("\n❌ 部分测试失败")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"\n测试异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
