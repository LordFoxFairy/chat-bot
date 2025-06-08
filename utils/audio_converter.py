import io
import pydub
from utils.logging_setup import logger
from typing import Optional
import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from data_models import AudioData, AudioFormat


def convert_to_target_format(
        audio_input: AudioData,
        target_sample_rate: int,
        target_channels: int,
        target_sample_width: int,  # 通常為 2 (16-bit)
        target_format_for_asr: str = "pcm_f32le"  # ASR期望的內部格式, pcm_f32le 指 float32 little-endian
) -> Optional[np.ndarray]:
    """
    將輸入的 AudioData 對象轉換為 ASR 模型期望的標準格式 (NumPy float32 數組)。

    參數:
        audio_input (AudioData): 包含原始音頻數據和格式信息的對象。
        target_sample_rate (int): 目標採樣率 (Hz)。
        target_channels (int): 目標通道数 (例如 1 代表單聲道)。
        target_sample_width (int): 目標樣本寬度 (字節，例如 2 代表 16-bit)。
        target_format_for_asr (str): 指示ASR模型期望的NumPy數組類型和格式。
                                     "pcm_f32le" 表示 float32 PCM。

    返回:
        Optional[np.ndarray]: 轉換後的 float32 NumPy 數組，如果轉換失敗則返回 None。
    """
    if not pydub:
        logger.error("音頻轉換失敗: 'pydub' 庫未安裝。請運行 'pip install pydub'。")
        return None
    if not AudioSegment or not CouldntDecodeError:
        logger.error("音頻轉換失敗: 'pydub' 庫組件未能正確導入。")
        return None
    if not audio_input.data:
        logger.warning("輸入的音頻數據為空，無法轉換。")
        return np.array([], dtype=np.float32)

    try:
        # 1. 從字節加載音頻數據
        # 如果是 PCM 格式，需要提供額外的元數據給 pydub
        if audio_input.format == AudioFormat.PCM:
            logger.debug(
                f"從 PCM 數據加載: sr={audio_input.sample_rate}, ch={audio_input.channels}, sw={audio_input.sample_width}")
            segment = AudioSegment(
                data=audio_input.data,
                sample_width=audio_input.sample_width,
                frame_rate=audio_input.sample_rate,
                channels=audio_input.channels
            )
        else:
            # 對於其他格式 (如 wav, mp3, opus)，pydub 可以從文件頭讀取元數據
            logger.debug(f"從文件格式 '{audio_input.format.value}' 加載音頻數據。")
            segment = AudioSegment.from_file(
                io.BytesIO(audio_input.data),
                format=audio_input.format.value
            )

        # 2. 轉換採樣率
        if segment.frame_rate != target_sample_rate:
            logger.debug(f"轉換採樣率: {segment.frame_rate} Hz -> {target_sample_rate} Hz")
            segment = segment.set_frame_rate(target_sample_rate)

        # 3. 轉換通道數
        if segment.channels != target_channels:
            logger.debug(f"轉換通道數: {segment.channels} -> {target_channels}")
            segment = segment.set_channels(target_channels)

        # 4. 轉換樣本寬度 (pydub 內部處理，導出為16-bit PCM 以便轉為 np.int16)
        if segment.sample_width != target_sample_width:
            logger.debug(f"轉換樣本寬度: {segment.sample_width} bytes -> {target_sample_width} bytes")
            segment = segment.set_sample_width(target_sample_width)

        # 5. 獲取原始 PCM 數據 (應為目標樣本寬度，例如 16-bit)
        pcm_data_target_width = segment.raw_data

        # 6. 轉換為 NumPy 數組 (基於目標樣本寬度)
        dtype_for_numpy = np.int16  # 假設 target_sample_width 為 2 (16-bit)
        if target_sample_width == 1:
            dtype_for_numpy = np.int8  # type: ignore
        elif target_sample_width == 4:  # 例如 32-bit int PCM
            dtype_for_numpy = np.int32  # type: ignore

        audio_np_intermediate = np.frombuffer(pcm_data_target_width, dtype=dtype_for_numpy)

        # 7. 如果 ASR 期望 float32，則進行轉換
        if target_format_for_asr == "pcm_f32le":
            # 將 int 類型數組轉換為 float32
            # 如果原始是 int16，除以 32768.0 進行歸一化 (可選，取決於模型期望)
            # FunASR SenseVoice 通常期望未歸一化的 float32，所以這裡不進行除法操作
            audio_np_float32 = audio_np_intermediate.astype(np.float32)
            logger.debug(f"成功將音頻轉換為 float32 NumPy 數組，形狀: {audio_np_float32.shape}")
            return audio_np_float32
        elif target_format_for_asr == "pcm_s16le" and target_sample_width == 2:
            logger.debug(f"成功將音頻轉換為 int16 NumPy 數組，形狀: {audio_np_intermediate.shape}")
            return audio_np_intermediate  # 已經是 int16
        else:
            logger.error(f"不支持的目標 ASR 格式 '{target_format_for_asr}' 或與目標樣本寬度不匹配。")
            return None

    except CouldntDecodeError as e:
        logger.error(f"無法解碼音頻數據 (格式: {audio_input.format.value}): {e}")
        return None
    except Exception as e:
        logger.error(f"音頻轉換過程中發生未知錯誤: {e}", exc_info=True)
        return None