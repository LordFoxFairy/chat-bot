import io
import pydub
from utils.logging_setup import logger
from typing import Optional
import numpy as np
from models import AudioData, AudioFormat

# 尝试导入 pydub，如果失败则使用 torchaudio
try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# 尝试导入 torchaudio 作为备选
try:
    import torchaudio
    import torch
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


def convert_audio_format_torchaudio(
        audio: AudioData,
        sample_rate: int,
        channels: int,
        output_format: str = "pcm_f32le"
) -> Optional[np.ndarray]:
    """使用 torchaudio 转换音频格式"""
    try:
        # 从字节加载音频
        audio_bytes = io.BytesIO(audio.data)
        waveform, orig_sample_rate = torchaudio.load(audio_bytes, format=audio.format.value)

        # 重采样
        if orig_sample_rate != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sample_rate, sample_rate)
            waveform = resampler(waveform)

        # 转换通道数
        if waveform.shape[0] != channels:
            if channels == 1:  # 转为单声道
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif channels == 2 and waveform.shape[0] == 1:  # 转为立体声
                waveform = waveform.repeat(2, 1)

        # 转换为 numpy
        audio_np = waveform.squeeze().numpy()

        if output_format == "pcm_f32le":
            return audio_np.astype(np.float32)
        else:
            # 转为 int16
            return (audio_np * 32767).astype(np.int16)

    except Exception as e:
        logger.error(f"torchaudio 音频转换失败: {e}", exc_info=True)
        return None


def convert_audio_format(
        audio: AudioData,
        sample_rate: int,
        channels: int,
        sample_width: int,  # 通常为 2 (16-bit)
        output_format: str = "pcm_f32le"  # ASR期望的内部格式, pcm_f32le 指 float32 little-endian
) -> Optional[np.ndarray]:
    """将音频转换为 ASR 模型需要的格式

    Args:
        audio: 输入音频数据
        sample_rate: 目标采样率 (Hz)
        channels: 目标通道数 (1=单声道)
        sample_width: 目标样本宽度 (字节，2=16bit)
        output_format: 输出格式 ("pcm_f32le" 或 "pcm_s16le")

    Returns:
        转换后的 NumPy 数组，失败返回 None
    """
    # 优先使用 torchaudio
    if TORCHAUDIO_AVAILABLE:
        logger.debug("使用 torchaudio 转换音频")
        return convert_audio_format_torchaudio(audio, sample_rate, channels, output_format)

    if not PYDUB_AVAILABLE:
        logger.error("音频转换失败: 'pydub' 和 'torchaudio' 库都未安装。")
        return None

    if not audio.data:
        logger.warning("输入的音频数据为空，无法转换。")
        return np.array([], dtype=np.float32)

    try:
        # 1. 从字节加载音频数据
        # 如果是 PCM 格式，需要提供额外的元数据给 pydub
        if audio.format == AudioFormat.PCM:
            logger.debug(
                f"从 PCM 数据加载: sr={audio.sample_rate}, ch={audio.channels}, sw={audio.sample_width}")
            segment = AudioSegment(
                data=audio.data,
                sample_width=audio.sample_width,
                frame_rate=audio.sample_rate,
                channels=audio.channels
            )
        else:
            # 对于其他格式 (如 wav, mp3, opus)，pydub 可以从文件头读取元数据
            logger.debug(f"从文件格式 '{audio.format.value}' 加载音频数据。")
            segment = AudioSegment.from_file(
                io.BytesIO(audio.data),
                format=audio.format.value
            )

        # 2. 转换采样率
        if segment.frame_rate != sample_rate:
            logger.debug(f"转换采样率: {segment.frame_rate} Hz -> {sample_rate} Hz")
            segment = segment.set_frame_rate(sample_rate)

        # 3. 转换通道数
        if segment.channels != channels:
            logger.debug(f"转换通道数: {segment.channels} -> {channels}")
            segment = segment.set_channels(channels)

        # 4. 转换样本宽度 (pydub 内部处理，导出为16-bit PCM 以便转为 np.int16)
        if segment.sample_width != sample_width:
            logger.debug(f"转换样本宽度: {segment.sample_width} bytes -> {sample_width} bytes")
            segment = segment.set_sample_width(sample_width)

        # 5. 获取原始 PCM 数据 (应为目标样本宽度，例如 16-bit)
        pcm_data = segment.raw_data

        # 6. 转换为 NumPy 数组 (基于目标样本宽度)
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sample_width, np.int16)
        audio_np = np.frombuffer(pcm_data, dtype=dtype)

        # 7. 如果 ASR 期望 float32，则进行转换
        if output_format == "pcm_f32le":
            # 将 int 类型数组转换为 float32
            # FunASR SenseVoice 通常期望未归一化的 float32，所以这里不进行除法操作
            audio_float32 = audio_np.astype(np.float32)
            logger.debug(f"成功将音频转换为 float32 NumPy 数组，形状: {audio_float32.shape}")
            return audio_float32
        elif output_format == "pcm_s16le" and sample_width == 2:
            logger.debug(f"成功将音频转换为 int16 NumPy 数组，形状: {audio_np.shape}")
            return audio_np  # 已经是 int16
        else:
            logger.error(f"不支持的目标 ASR 格式 '{output_format}' 或与目标样本宽度不匹配。")
            return None

    except CouldntDecodeError as e:
        logger.error(f"无法解码音频数据 (格式: {audio.format.value}): {e}")
        return None
    except Exception as e:
        logger.error(f"音频转换过程中发生未知错误: {e}", exc_info=True)
        return None
