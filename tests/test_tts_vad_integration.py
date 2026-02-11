"""
端到端测试：TTS 生成语音 -> VAD 检测语音活动

测试步骤：
1. 使用 EdgeTTSAdapter 将文字 "你好，我是语音助手" 转换为音频
2. 将生成的音频传给 SileroVADAdapter 进行语音活动检测
3. 验证 VAD 能正确检测到语音（is_speech=True）

运行方式：
    cd /Users/nako/PycharmProjects/github/thefoxfairy/chat-bot
    export PATH=$PATH:/opt/homebrew/bin  # 确保 ffmpeg 在 PATH 中
    export PYTHONPATH=$PYTHONPATH:.
    python3 tests/test_tts_vad_integration.py
"""
import asyncio
import io
import os
import subprocess
import sys

import numpy as np
import torch

from backend.adapters.tts.edge_tts_adapter import EdgeTTSAdapter
from backend.adapters.vad.silero_vad_adapter import SileroVADAdapter
from backend.core.models import TextData


def convert_mp3_to_pcm(mp3_bytes: bytes, target_sr: int = 16000) -> bytes:
    """将 MP3 音频转换为 PCM 格式 (int16, mono)

    尝试使用 pydub (需要 ffmpeg) 或 subprocess 调用 ffmpeg
    """
    # 方案 1: 尝试 pydub
    try:
        from pydub import AudioSegment

        audio_f = io.BytesIO(mp3_bytes)
        segment = AudioSegment.from_mp3(audio_f)

        # 重采样
        if segment.frame_rate != target_sr:
            segment = segment.set_frame_rate(target_sr)

        # 转单声道
        if segment.channels > 1:
            segment = segment.set_channels(1)

        # 确保 16bit
        if segment.sample_width != 2:
            segment = segment.set_sample_width(2)

        return segment.raw_data
    except Exception:
        pass

    # 方案 2: 使用 ffmpeg 命令行
    try:
        tmp_mp3 = "/tmp/tts_test_output.mp3"
        tmp_wav = "/tmp/tts_test_output.wav"

        with open(tmp_mp3, "wb") as f:
            f.write(mp3_bytes)

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", str(target_sr), "-ac", "1", tmp_wav],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.exists(tmp_wav):
            import soundfile as sf

            data, _ = sf.read(tmp_wav)
            audio_int16 = (data * 32767).astype(np.int16)
            return audio_int16.tobytes()
    except Exception:
        pass

    return None


async def test_tts_to_vad():
    """
    端到端测试：TTS 生成语音 -> VAD 检测语音活动
    """
    print("\n" + "=" * 60)
    print("端到端测试：TTS 生成语音 -> VAD 检测语音活动")
    print("=" * 60 + "\n")

    # 1. 准备配置
    tts_config = {
        "voice": "zh-CN-XiaoxiaoNeural",
        "rate": "+0%",
        "volume": "+0%",
        "save_generated_audio": False,
    }

    vad_config = {
        "model_repo_path": "outputs/models/vad/silero-vad",
        "model_name": "silero_vad",
        "threshold": 0.5,
        "device": "cpu",
        "window_size_samples": 512,
        "sample_rate": 16000,
    }

    # 2. 初始化适配器
    print("[步骤 1/4] 初始化适配器...")
    tts_adapter = EdgeTTSAdapter("tts-test", tts_config)
    vad_adapter = SileroVADAdapter("vad-test", vad_config)

    test_passed = False

    try:
        await tts_adapter.setup()
        await vad_adapter.setup()
        print("[OK] 适配器初始化成功")

        # 3. TTS 生成语音
        text = "你好，我是语音助手"
        print(f"\n[步骤 2/4] 执行 TTS: '{text}'...")

        text_data = TextData(text=text)
        audio_chunks = []

        async for chunk in tts_adapter.synthesize_stream(text_data):
            if not chunk.is_final and chunk.data:
                audio_chunks.append(chunk.data)

        full_audio = b"".join(audio_chunks)
        print(f"[OK] TTS 生成完成，音频大小: {len(full_audio)} 字节")

        # 4. 音频格式转换
        print("\n[步骤 3/4] 音频预处理 (MP3 -> PCM)...")

        audio_bytes = convert_mp3_to_pcm(full_audio, target_sr=16000)

        if audio_bytes is None:
            print("\n[ERROR] 无法转换 MP3 音频")
            print("        请确保安装了 ffmpeg: brew install ffmpeg")
            return False

        print(f"[OK] 音频转换完成，PCM 数据大小: {len(audio_bytes)} 字节")

        # 5. VAD 检测
        print("\n[步骤 4/4] 执行 VAD 检测...")

        # Silero VAD 需要按窗口大小处理
        window_size_bytes = 512 * 2  # 512 samples * 2 bytes/sample (int16)

        speech_detected = False
        chunks_with_speech = 0
        total_chunks = 0
        probs = []

        # 重置 VAD 状态
        await vad_adapter.reset_state()

        for i in range(0, len(audio_bytes), window_size_bytes):
            chunk = audio_bytes[i : i + window_size_bytes]

            # 如果最后一块不足 window_size，填充零
            if len(chunk) < window_size_bytes:
                chunk = chunk + b"\x00" * (window_size_bytes - len(chunk))

            # 调用 VAD 检测
            is_speech = await vad_adapter.detect(chunk)

            # 记录概率用于调试
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32).to(vad_adapter.device)
            with torch.no_grad():
                speech_prob = vad_adapter.model(
                    audio_tensor, vad_adapter.sample_rate
                ).item()
                probs.append(speech_prob)

            total_chunks += 1

            if is_speech:
                speech_detected = True
                chunks_with_speech += 1

        print(f"\n检测统计:")
        print(f"  - 总块数: {total_chunks}")
        print(f"  - 语音块数: {chunks_with_speech}")
        print(
            f"  - 概率范围: min={min(probs):.4f}, max={max(probs):.4f}, avg={np.mean(probs):.4f}"
        )
        print(f"  - 使用阈值: {vad_adapter.threshold}")

        # 验证结果
        print("\n" + "-" * 40)
        if speech_detected:
            print("[PASS] VAD 成功检测到语音活动 (is_speech=True)")
            print(f"       语音占比: {chunks_with_speech/total_chunks:.1%}")
            test_passed = True
        else:
            print("[FAIL] VAD 未检测到语音活动")
            low_threshold = 0.3
            low_speech_count = sum(1 for p in probs if p > low_threshold)
            if low_speech_count > 0:
                print(
                    f"       使用阈值 {low_threshold} 可检测到 {low_speech_count}/{total_chunks} 块"
                )
            test_passed = False

    except Exception as e:
        print(f"\n[ERROR] 测试过程中出错: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await tts_adapter.close()
        await vad_adapter.close()
        print("\n" + "=" * 60)

    return test_passed


if __name__ == "__main__":
    try:
        result = asyncio.run(test_tts_to_vad())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        pass
