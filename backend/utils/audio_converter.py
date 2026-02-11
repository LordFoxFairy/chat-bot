import io
from typing import Optional

import numpy as np
from scipy import signal

from backend.core.constants import (
    HIGH_PASS_CUTOFF_HZ,
    LOW_PASS_CUTOFF_HZ,
    NOISE_GATE_THRESHOLD_RATIO,
    NOISE_GATE_WINDOW_MS,
)
from backend.core.models import AudioData, AudioFormat
from backend.utils.logging_setup import logger


def apply_noise_reduction(audio_np: np.ndarray, sample_rate: int) -> np.ndarray:
    """应用基础噪声抑制处理

    Args:
        audio_np: 输入音频 numpy 数组 (int16 或 float32)
        sample_rate: 采样率

    Returns:
        处理后的音频 numpy 数组
    """
    original_dtype = audio_np.dtype

    # 转换为 float64 进行处理
    if audio_np.dtype == np.int16:
        audio_float = audio_np.astype(np.float64) / 32768.0
    elif audio_np.dtype in [np.float32, np.float64]:
        audio_float = audio_np.astype(np.float64)
    else:
        return audio_np  # 不支持的类型，直接返回

    # 1. 高通滤波器 - 移除低频噪声
    try:
        nyquist = sample_rate / 2
        high_cutoff = HIGH_PASS_CUTOFF_HZ / nyquist
        if high_cutoff < 1.0:  # 确保截止频率有效
            b, a = signal.butter(2, high_cutoff, btype='high')
            audio_float = signal.filtfilt(b, a, audio_float)
    except Exception as e:
        logger.debug(f"高通滤波器应用失败: {e}")

    # 2. 低通滤波器 - 移除高频噪声
    try:
        low_cutoff = min(LOW_PASS_CUTOFF_HZ, sample_rate / 2 - 100) / nyquist
        if 0 < low_cutoff < 1.0:
            b, a = signal.butter(2, low_cutoff, btype='low')
            audio_float = signal.filtfilt(b, a, audio_float)
    except Exception as e:
        logger.debug(f"低通滤波器应用失败: {e}")

    # 3. 简单噪声门控 - 抑制低能量片段
    try:
        rms = np.sqrt(np.mean(audio_float ** 2))
        noise_gate_threshold = rms * NOISE_GATE_THRESHOLD_RATIO
        mask = np.abs(audio_float) > noise_gate_threshold
        # 平滑 mask 以避免突变
        kernel_size = min(int(sample_rate * NOISE_GATE_WINDOW_MS / 1000), 100)
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            mask_float = mask.astype(np.float64)
            mask_smooth = np.convolve(mask_float, kernel, mode='same')
            audio_float = audio_float * np.clip(mask_smooth, 0.1, 1.0)  # 保留至少 10%
    except Exception as e:
        logger.debug(f"噪声门控应用失败: {e}")

    # 转换回原始类型
    if original_dtype == np.int16:
        audio_float = np.clip(audio_float, -1.0, 1.0)
        return (audio_float * 32767).astype(np.int16)
    elif original_dtype == np.float32:
        return audio_float.astype(np.float32)
    else:
        return audio_float

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
        output_format: str = "pcm_f32le",
        raise_on_error: bool = False
) -> Optional[np.ndarray]:
    """使用 torchaudio 转换音频格式"""
    try:
        # 从字节加载音频
        audio_bytes = io.BytesIO(audio.data)

        # torchaudio 2.1+ requires torchcodec for load function if backend is not specified
        # but macOS + py3.13 environment might have issues
        # Let's try specifying backend or fallback to pydub if torchcodec missing
        try:
            waveform, orig_sample_rate = torchaudio.load(audio_bytes, format=audio.format.value)
        except ImportError as e:
            if "TorchCodec" in str(e):
                # If torchcodec is missing, try to force a different backend or just fail and let fallback handle it
                # if sox is available, maybe use sox_io?
                # For now, let's just raise so we fallback to pydub if available
                # But wait, logic below says "if TORCHAUDIO_AVAILABLE: return ... " and doesn't fallback to pydub
                # We should modify the main function to handle this fallback
                raise
            raise

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

        # 应用噪声抑制处理
        try:
            audio_np = apply_noise_reduction(audio_np.astype(np.float32), sample_rate)
            logger.debug("torchaudio: 已应用噪声抑制处理")
        except Exception as e:
            logger.warning(f"torchaudio: 噪声抑制处理失败，使用原始音频: {e}")

        if output_format == "pcm_f32le":
            # torchaudio 输出已经是归一化的 [-1, 1] 范围
            return audio_np.astype(np.float32)
        else:
            # 转为 int16
            audio_np = np.clip(audio_np, -1.0, 1.0)
            return (audio_np * 32767).astype(np.int16)

    except Exception as e:
        logger.error(f"torchaudio 音频转换失败: {e}")
        if raise_on_error:
            raise
        return None


def _load_audio_segment(audio: AudioData) -> "AudioSegment":
    """从 AudioData 加载 pydub AudioSegment

    Args:
        audio: 输入音频数据

    Returns:
        pydub AudioSegment 对象
    """
    if audio.format == AudioFormat.PCM:
        logger.debug(
            f"从 PCM 数据加载: sr={audio.sample_rate}, ch={audio.channels}, sw={audio.sample_width}")
        return AudioSegment(
            data=audio.data,
            sample_width=audio.sample_width,
            frame_rate=audio.sample_rate,
            channels=audio.channels
        )
    else:
        logger.debug(f"从文件格式 '{audio.format.value}' 加载音频数据。")
        return AudioSegment.from_file(
            io.BytesIO(audio.data),
            format=audio.format.value
        )


def _apply_audio_transformations(
        segment: "AudioSegment",
        sample_rate: int,
        channels: int,
        sample_width: int
) -> "AudioSegment":
    """应用采样率、通道数、样本宽度转换

    Args:
        segment: pydub AudioSegment 对象
        sample_rate: 目标采样率
        channels: 目标通道数
        sample_width: 目标样本宽度

    Returns:
        转换后的 AudioSegment 对象
    """
    if segment.frame_rate != sample_rate:
        logger.debug(f"转换采样率: {segment.frame_rate} Hz -> {sample_rate} Hz")
        segment = segment.set_frame_rate(sample_rate)

    if segment.channels != channels:
        logger.debug(f"转换通道数: {segment.channels} -> {channels}")
        segment = segment.set_channels(channels)

    if segment.sample_width != sample_width:
        logger.debug(f"转换样本宽度: {segment.sample_width} bytes -> {sample_width} bytes")
        segment = segment.set_sample_width(sample_width)

    return segment


def _segment_to_numpy(segment: "AudioSegment", sample_width: int) -> np.ndarray:
    """将 AudioSegment 转换为 NumPy 数组

    Args:
        segment: pydub AudioSegment 对象
        sample_width: 样本宽度（字节）

    Returns:
        NumPy 数组
    """
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width, np.int16)
    return np.frombuffer(segment.raw_data, dtype=dtype)


def _convert_to_output_format(
        audio_np: np.ndarray,
        output_format: str,
        sample_width: int,
        raise_on_error: bool = False
) -> Optional[np.ndarray]:
    """将 NumPy 数组转换为目标输出格式

    Args:
        audio_np: 输入 NumPy 数组
        output_format: 输出格式 ("pcm_f32le" 或 "pcm_s16le")
        sample_width: 样本宽度
        raise_on_error: 发生错误时是否抛出异常

    Returns:
        转换后的 NumPy 数组
    """
    if output_format == "pcm_f32le":
        if audio_np.dtype == np.int16:
            audio_float32 = audio_np.astype(np.float32) / 32768.0
        else:
            audio_float32 = audio_np.astype(np.float32)
        logger.debug(f"成功将音频转换为归一化 float32 NumPy 数组，形状: {audio_float32.shape}")
        return audio_float32
    elif output_format == "pcm_s16le" and sample_width == 2:
        logger.debug(f"成功将音频转换为 int16 NumPy 数组，形状: {audio_np.shape}")
        return audio_np
    else:
        error_msg = f"不支持的目标 ASR 格式 '{output_format}' 或与目标样本宽度不匹配。"
        logger.error(error_msg)
        if raise_on_error:
            raise ValueError(error_msg)
        return None


def _convert_with_pydub(
        audio: AudioData,
        sample_rate: int,
        channels: int,
        sample_width: int,
        output_format: str,
        raise_on_error: bool = False
) -> Optional[np.ndarray]:
    """使用 pydub 转换音频格式

    Args:
        audio: 输入音频数据
        sample_rate: 目标采样率
        channels: 目标通道数
        sample_width: 目标样本宽度
        output_format: 输出格式
        raise_on_error: 发生错误时是否抛出异常

    Returns:
        转换后的 NumPy 数组
    """
    try:
        # 加载音频
        segment = _load_audio_segment(audio)

        # 应用转换
        segment = _apply_audio_transformations(segment, sample_rate, channels, sample_width)

        # 转换为 NumPy
        audio_np = _segment_to_numpy(segment, sample_width)

        # 应用噪声抑制
        try:
            audio_np = apply_noise_reduction(audio_np, sample_rate)
            logger.debug("已应用噪声抑制处理")
        except Exception as e:
            logger.warning(f"噪声抑制处理失败，使用原始音频: {e}")

        # 转换为目标格式
        return _convert_to_output_format(audio_np, output_format, sample_width, raise_on_error)

    except CouldntDecodeError as e:
        logger.error(f"无法解码音频数据 (格式: {audio.format.value}): {e}")
        if raise_on_error:
            raise
        return None
    except Exception as e:
        logger.error(f"音频转换过程中发生未知错误: {e}", exc_info=True)
        if raise_on_error:
            raise
        return None


def convert_audio_format(
        audio: AudioData,
        sample_rate: int,
        channels: int,
        sample_width: int,
        output_format: str = "pcm_f32le",
        raise_on_error: bool = False
) -> Optional[np.ndarray]:
    """将音频转换为 ASR 模型需要的格式

    这是主入口函数，优先使用 torchaudio，失败后降级到 pydub。

    Args:
        audio: 输入音频数据
        sample_rate: 目标采样率 (Hz)
        channels: 目标通道数 (1=单声道)
        sample_width: 目标样本宽度 (字节，2=16bit)
        output_format: 输出格式 ("pcm_f32le" 或 "pcm_s16le")
        raise_on_error: 发生错误时是否抛出异常

    Returns:
        转换后的 NumPy 数组，失败返回 None (除非 raise_on_error=True)
    """
    # 优先使用 torchaudio
    if TORCHAUDIO_AVAILABLE:
        logger.debug("尝试使用 torchaudio 转换音频")
        result = convert_audio_format_torchaudio(
            audio, sample_rate, channels, output_format, raise_on_error=False
        )
        if result is not None:
            return result
        logger.warning("torchaudio 转换失败，尝试降级到 pydub...")

    # 降级到 pydub
    if not PYDUB_AVAILABLE:
        error_msg = "音频转换失败: 'pydub' 和 'torchaudio' 库都未安装。"
        logger.error(error_msg)
        if raise_on_error:
            raise RuntimeError(error_msg)
        return None

    # 空数据检查
    if not audio.data:
        logger.warning("输入的音频数据为空，无法转换。")
        return np.array([], dtype=np.float32)

    return _convert_with_pydub(
        audio, sample_rate, channels, sample_width, output_format, raise_on_error
    )
