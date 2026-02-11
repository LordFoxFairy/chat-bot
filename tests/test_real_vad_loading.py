import asyncio
import os
import sys
import yaml
import numpy as np
import torch
import logging
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.append(os.getcwd())

from backend.adapters.vad.silero_vad_adapter import SileroVADAdapter
from backend.utils.logging_setup import setup_logging
from backend.core.config_models import AppConfig

async def test_vad_loading():
    print("--- 1. 读取配置文件 ---")
    config_path = Path("backend/configs/config.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    # 初始化日志
    setup_logging(config_dict.get("logging", {}))

    # 获取 VAD 配置
    # 注意：config.yaml 结构是 modules -> vad -> config -> silero_vad
    vad_config_all = config_dict.get("modules", {}).get("vad", {})
    if not vad_config_all.get("enabled"):
        print("VAD module is disabled in config.")
        return

    adapter_type = vad_config_all.get("adapter_type")
    # 具体配置在 config -> silero_vad 下
    vad_specific_config = vad_config_all.get("config", {}).get(adapter_type, {})

    # 合并 top-level config (like threshold) if needed, but SileroVADAdapter reads from its own config dict
    # BaseVAD reads sample_rate and threshold from the passed config dict
    # So we need to ensure the config dict passed to the adapter has all necessary fields

    print(f"VAD Config: {vad_specific_config}")

    print("\n--- 2. 创建并初始化 SileroVADAdapter ---")
    try:
        vad_adapter = SileroVADAdapter(
            module_id="vad_test",
            config=vad_specific_config
        )

        print("Executing setup()...")
        await vad_adapter.setup()
        print("Setup complete.")

    except Exception as e:
        print(f"Failed to initialize VAD adapter: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- 3. 验证模型加载 ---")
    if vad_adapter.model is not None:
        print("Model loaded successfully.")
        print(f"Model type: {type(vad_adapter.model)}")
        print(f"Device: {vad_adapter.device}")
    else:
        print("Error: Model is None after setup.")
        return

    print("\n--- 4. 测试基本的 VAD 处理功能 (静音测试) ---")

    # 创建 1秒钟的静音数据 (16kHz, 16bit mono)
    sample_rate = vad_adapter.sample_rate # Should be 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    # 生成静音 (全0) - int16 bytes
    silence_audio_np = np.zeros(num_samples, dtype=np.int16)
    silence_audio_bytes = silence_audio_np.tobytes()

    print(f"Testing with {duration_sec}s silence ({len(silence_audio_bytes)} bytes)...")

    try:
        start_time = asyncio.get_event_loop().time()
        result = await vad_adapter.detect(silence_audio_bytes)
        end_time = asyncio.get_event_loop().time()

        print(f"Detection result: {result}")
        print(f"Time taken: {(end_time - start_time)*1000:.2f} ms")

        if result is False:
            print("SUCCESS: Silence correctly detected as non-speech.")
        else:
            print("FAILURE: Silence detected as speech!")

    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

    # 可选：测试一段模拟语音（高斯噪声有时会被误判为语音，或者用正弦波）
    # Silero VAD 对噪音很鲁棒，纯噪音可能还是 False。
    # 这里只按照要求做基本的静音测试。

    print("\n--- 清理资源 ---")
    await vad_adapter.close()
    print("Test finished.")

if __name__ == "__main__":
    asyncio.run(test_vad_loading())
