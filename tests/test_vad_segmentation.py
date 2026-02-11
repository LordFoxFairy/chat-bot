"""
VAD 分段测试：模拟连续语音流，验证 VAD 正确分段

测试步骤：
1. 用 TTS 生成两段语音："你好", "再见"
2. 在两段语音之间插入 1 秒静音
3. 将合并后的音频分成小块（512 samples）发送给 VAD
4. 验证 VAD 能检测到：
   - 第一段语音开始
   - 中间静音（语音结束）
   - 第二段语音开始
   - 最后静音（语音结束）

使用真实代码，不进行 Mock。
"""
import asyncio
import io
import os
import subprocess
import sys
import time

import numpy as np
import torch

from backend.adapters.tts.edge_tts_adapter import EdgeTTSAdapter
from backend.adapters.vad.silero_vad_adapter import SileroVADAdapter
from backend.core.models import TextData


def convert_mp3_to_pcm(mp3_bytes: bytes, target_sr: int = 16000) -> bytes:
    """将 MP3 音频转换为 PCM 格式 (int16, mono)"""
    # 尝试使用 pydub
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
    except ImportError:
        pass
    except Exception as e:
        print(f"Pydub 转换失败: {e}")

    # 回退到 ffmpeg
    try:
        tmp_mp3 = f"/tmp/vad_segment_test_{int(time.time())}.mp3"
        tmp_wav = f"/tmp/vad_segment_test_{int(time.time())}.wav"

        with open(tmp_mp3, "wb") as f:
            f.write(mp3_bytes)

        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_wav],
            capture_output=True, check=True
        )

        with open(tmp_wav, "rb") as f:
            # 跳过 WAV 头 (44字节)
            f.seek(44)
            return f.read()

    except Exception as e:
        print(f"FFmpeg 转换失败: {e}")
        return None
    finally:
        if os.path.exists(tmp_mp3): os.remove(tmp_mp3)
        if os.path.exists(tmp_wav): os.remove(tmp_wav)


async def generate_speech(tts_adapter: EdgeTTSAdapter, text: str) -> bytes:
    """生成一段语音的 PCM 数据"""
    print(f"  正在生成: '{text}'...")
    text_data = TextData(text=text)
    audio_chunks = []

    async for chunk in tts_adapter.synthesize_stream(text_data):
        if not chunk.is_final and chunk.data:
            audio_chunks.append(chunk.data)

    full_audio = b"".join(audio_chunks)
    pcm_data = convert_mp3_to_pcm(full_audio)

    if not pcm_data:
        raise RuntimeError(f"音频转换失败: {text}")

    print(f"  生成的 PCM 数据大小: {len(pcm_data)} 字节")
    return pcm_data


async def run_vad_segmentation_test():
    print("\n" + "=" * 60)
    print("VAD 分段测试：验证连续语音流分段能力")
    print("=" * 60 + "\n")

    # 1. 初始化配置
    tts_config = {
        "voice": "zh-CN-XiaoxiaoNeural",
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

    tts_adapter = EdgeTTSAdapter("tts-test", tts_config)
    vad_adapter = SileroVADAdapter("vad-test", vad_config)

    try:
        await tts_adapter.setup()
        await vad_adapter.setup()
        print("[OK] 适配器初始化成功")

        # 2. 准备测试音频数据
        print("\n[STEP 1] 准备测试音频流...")

        # 生成两个语音片段
        speech1 = await generate_speech(tts_adapter, "你好")
        speech2 = await generate_speech(tts_adapter, "再见")

        # 生成静音片段 (1秒)
        sample_rate = 16000
        sample_width = 2 # 16-bit
        silence_duration = 1.0
        silence_bytes = int(sample_rate * silence_duration * sample_width)
        silence = b'\x00' * silence_bytes
        print(f"  生成的静音数据大小: {len(silence)} 字节 ({silence_duration}s)")

        # 拼接音频流: 0.5s静音 + 语音1 + 1s静音 + 语音2 + 0.5s静音
        pre_silence = b'\x00' * int(sample_rate * 0.5 * sample_width)
        post_silence = b'\x00' * int(sample_rate * 0.5 * sample_width)

        full_stream = pre_silence + speech1 + silence + speech2 + post_silence
        print(f"\n[OK] 音频流拼接完成，总大小: {len(full_stream)} 字节 (约 {len(full_stream)/(16000*2):.2f}s)")

        # 3. 运行 VAD 检测
        print("\n[STEP 2] 运行流式 VAD 检测...")

        window_size_samples = 512
        window_size_bytes = window_size_samples * 2

        states = [] # 记录每一帧的状态 (is_speech, prob)

        await vad_adapter.reset_state()

        for i in range(0, len(full_stream), window_size_bytes):
            chunk = full_stream[i : i + window_size_bytes]

            # 填充最后一块
            if len(chunk) < window_size_bytes:
                chunk = chunk + b"\x00" * (window_size_bytes - len(chunk))

            is_speech = await vad_adapter.detect(chunk)

            # 获取内部概率用于分析
            # 注意：实际生产中无法直接访问 model 和 internal state，这里仅用于验证测试
            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32).to(vad_adapter.device)
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            with torch.no_grad():
                prob = vad_adapter.model(audio_tensor, vad_adapter.sample_rate).item()

            states.append({
                "time": i / (sample_rate * 2), # 时间戳
                "is_speech": is_speech,
                "prob": prob
            })

        print(f"[OK] VAD 处理完成，共处理 {len(states)} 帧")

        # 4. 分析结果
        print("\n[STEP 3] 分析检测结果...")

        # 寻找状态转换点
        transitions = []
        current_state = False

        # 使用简单的状态机来检测语音段
        # 为了避免抖动，我们可以使用简单的平滑逻辑
        # 但这里的目标是验证 VAD 的输出，所以直接看 transitions

        speech_segments = []
        segment_start = -1

        for state in states:
            if state["is_speech"] and segment_start == -1:
                segment_start = state["time"]
            elif not state["is_speech"] and segment_start != -1:
                duration = state["time"] - segment_start
                if duration > 0.1: # 忽略太短的杂音
                    speech_segments.append((segment_start, state["time"]))
                segment_start = -1

        # 处理最后一段
        if segment_start != -1:
            speech_segments.append((segment_start, states[-1]["time"]))

        print(f"检测到的语音片段数: {len(speech_segments)}")
        for i, (start, end) in enumerate(speech_segments):
            print(f"  片段 {i+1}: {start:.2f}s - {end:.2f}s (时长: {end-start:.2f}s)")

        # 5. 验证是否符合预期
        print("\n[STEP 4] 结果验证")

        # 预期：应该有两个明显的两段语音
        # "你好" 虽然短，但应该 > 0.3s
        # "再见" 也是
        # 并且两段之间应该有间隔

        if len(speech_segments) != 2:
            print(f"[FAIL] 预期检测到 2 个语音片段，实际检测到 {len(speech_segments)} 个")

            # 打印详细的时间轴
            print("\n详细概率时间轴:")
            for i in range(0, len(states), 10): # 每10帧打印一次
                s = states[i]
                marker = "#" if s["is_speech"] else "."
                print(f"{s['time']:.2f}s [{marker}] prob={s['prob']:.4f}")

            return False

        # 检查间隔
        gap = speech_segments[1][0] - speech_segments[0][1]
        print(f"检测到的语音间隔: {gap:.2f}s (预期约 1.0s)")

        if gap < 0.5:
            print(f"[FAIL] 语音间隔太短 ({gap:.2f}s)，未能正确分段")
            return False

        print("[PASS] VAD 成功将连续语音流正确分段！")
        return True

    except Exception as e:
        print(f"[ERROR] 测试发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await tts_adapter.close()
        await vad_adapter.close()

if __name__ == "__main__":
    try:
        success = asyncio.run(run_vad_segmentation_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        pass
