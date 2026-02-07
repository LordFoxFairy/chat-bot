"""VAD Adapter 单元测试"""
import asyncio
import os
import sys
import numpy as np

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logging_setup import logger
from src.utils.config_loader import ConfigLoader
from src.adapters.vad.silero_vad_adapter import SileroVADAdapter


async def test_silero_vad():
    """测试 Silero VAD Adapter"""
    logger.info("\n===== 测试 Silero VAD Adapter =====")

    # 加载配置
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'config.yaml')
    logger.info(f"加载配置: {config_path}")

    try:
        config = await ConfigLoader.load_config(config_path)
        if not config:
            logger.error("配置加载失败")
            return False
    except Exception as e:
        logger.error(f"配置加载错误: {e}", exc_info=True)
        return False

    # 获取 VAD 配置
    vad_config = config.get("modules", {}).get("vad")
    if not vad_config:
        logger.error("配置中未找到 'modules.vad'")
        return False

    try:
        # 创建 VAD 实例
        logger.info("创建 SileroVADAdapter 实例...")
        vad = SileroVADAdapter(
            module_id="test_vad",
            config=vad_config
        )

        # 初始化
        logger.info("初始化 VAD...")
        await vad.setup()

        if not vad.is_ready:
            logger.error("VAD 初始化后未就绪")
            return False

        logger.info("✓ VAD 初始化成功")

        # 测试检测语音（模拟音频数据）
        logger.info("测试语音检测...")

        # 生成测试音频（16kHz, 单声道, 100ms）
        sample_rate = 16000
        duration_ms = 100
        num_samples = int(sample_rate * duration_ms / 1000)

        # 静音音频
        silence_audio = np.zeros(num_samples, dtype=np.float32)
        is_speech_silence = await vad.detect_speech(silence_audio.tobytes())
        logger.info(f"静音检测: is_speech={is_speech_silence} (期望 False)")

        # 模拟语音音频（随机噪声）
        speech_audio = np.random.randn(num_samples).astype(np.float32) * 0.1
        is_speech_noise = await vad.detect_speech(speech_audio.tobytes())
        logger.info(f"噪声检测: is_speech={is_speech_noise}")

        logger.info("✓ 语音检测测试完成")

        # 停止
        if hasattr(vad, 'stop'):
            await vad.stop()
            logger.info("✓ VAD 已停止")

        logger.info("✅ Silero VAD Adapter 测试通过\n")
        return True

    except Exception as e:
        logger.error(f"VAD 测试失败: {e}", exc_info=True)
        return False


async def main():
    """运行 VAD 测试"""
    logger.info("\n" + "="*60)
    logger.info("开始 VAD Adapter 测试")
    logger.info("="*60)

    try:
        success = await test_silero_vad()

        if success:
            logger.info("\n" + "="*60)
            logger.info("✅ VAD 测试全部通过")
            logger.info("="*60 + "\n")
        else:
            logger.error("\n❌ VAD 测试失败")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"\n测试异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
