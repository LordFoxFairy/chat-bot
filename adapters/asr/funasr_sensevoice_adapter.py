import asyncio
from utils.logging_setup import logger
import os
from functools import partial  # 導入 partial
from typing import Optional, Dict, Any

import numpy as np

# 框架核心組件的相對路徑導入
from core.exceptions import ModuleInitializationError, ModuleProcessingError
# 導入框架定義的核心數據模型
# 導入重構後的 ASR 基類
from modules.base_asr import BaseASR  # 確保路徑正確

# 日誌記錄器


# 動態導入 FunASR
try:
    from funasr import AutoModel

    FUNASR_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning("funasr 庫未安裝，FunASRSenseVoiceAdapter 將無法使用。請運行 'pip install funasr'")
    AutoModel = None  # type: ignore
    FUNASR_AVAILABLE = False


class FunASRSenseVoiceAdapter(BaseASR):
    """
    FunASR SenseVoice ASR 適配器。
    實現 BaseASR 定義的 _infer_and_parse_audio_numpy 核心方法。
    嚴格根據傳入的 config 加載配置。
    """

    def __init__(self, module_id: str, config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 ):
        super().__init__(module_id, config, event_loop)

        if not FUNASR_AVAILABLE:  # pragma: no cover
            raise ModuleInitializationError(
                f"FunASR [{self.module_id}]: 'funasr' 庫未安裝。"
            )

        self.model_dir: str = self.adapter_specific_config.get("model_dir", "models/asr/SenseVoiceSmall")
        self.device: str = self.adapter_specific_config.get("device", "cpu")
        self.output_dir: Optional[str] = self.adapter_specific_config.get("output_dir")
        self.model_revision: Optional[str] = self.adapter_specific_config.get("model_revision")

        self.vad_chunk_size: Optional[int] = self.adapter_specific_config.get("vad_chunk_size")
        self.encoder_chunk_look_forward: Optional[int] = self.adapter_specific_config.get("encoder_chunk_look_forward")
        self.decoder_chunk_look_back: Optional[int] = self.adapter_specific_config.get("decoder_chunk_look_back")

        self.model: Optional[AutoModel] = None

        logger.info(f"FunASR [{self.module_id}] 適配器特定配置已從 BaseASR.adapter_specific_config 加載:")
        logger.info(f"  - model_dir: {self.model_dir}")
        logger.info(f"  - device: {self.device}")

    async def initialize(self):
        """異步初始化 FunASR SenseVoice 模型。"""
        logger.info(f"FunASR [{self.module_id}]: 正在初始化 SenseVoice 模型...")
        try:
            model_path = os.path.abspath(self.model_dir)
            if not os.path.exists(model_path):
                raise ModuleInitializationError(
                    f"FunASR [{self.module_id}]: 模型目錄 '{model_path}' 未找到。"
                )

            logger.info(f"FunASR [{self.module_id}]: 正在從 {model_path} 加載模型到設備 {self.device}...")

            params_for_model = {
                "model": model_path,
                "model_revision": self.model_revision,
                "device": self.device,
                "disable_pbar": True,
                "chunk_size": [self.vad_chunk_size if self.vad_chunk_size else 5, 10, 5],
                "encoder_chunk_look_forward": self.encoder_chunk_look_forward if self.encoder_chunk_look_forward is not None else 0,
                "decoder_chunk_look_back": self.decoder_chunk_look_back if self.decoder_chunk_look_back is not None else 0,
            }
            if self.output_dir:
                params_for_model["output_dir"] = os.path.abspath(self.output_dir)

            final_model_params = {k: v for k, v in params_for_model.items() if v is not None}
            logger.debug(f"FunASR [{self.module_id}]: 傳遞給 AutoModel 的最終參數: {final_model_params}")

            self.model = AutoModel(**final_model_params)  # type: ignore

            if self.model is None:  # pragma: no cover
                raise ModuleInitializationError(f"FunASR [{self.module_id}]: 從 {model_path} 加載模型失敗。")

            logger.info(f"FunASR [{self.module_id}]: SenseVoice 模型已成功從 '{model_path}' 加載。")
            self._is_initialized = True
            self._is_ready = True
        except Exception as e:  # pragma: no cover
            self._is_initialized = False
            self._is_ready = False
            logger.error(f"FunASR [{self.module_id}]: SenseVoice 模型初始化失敗: {e}", exc_info=True)
            raise ModuleInitializationError(f"FunASR [{self.module_id}] SenseVoice 模型初始化錯誤: {e}") from e

    async def _infer_and_parse_audio_numpy(self, audio_np: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        【實現】核心方法：執行 FunASR 模型推理並返回識別出的文本字符串。
        """
        log_prefix = f"FunASR [{self.module_id}] [_infer_and_parse_audio_numpy]"

        if self.model is None:  # pragma: no cover
            logger.error(f"{log_prefix} 被調用但模型未初始化。")
            raise ModuleProcessingError(f"FunASR [{self.module_id}] 模型未初始化。")

        if audio_np.size == 0:
            logger.debug(f"{log_prefix} 輸入空的 NumPy 音頻數組，返回 None。")
            return None

            # 1. 執行模型推理
        model_output: Any = None
        try:
            loop = asyncio.get_running_loop()

            # 使用 functools.partial 來正確傳遞參數給 self.model.generate
            # self.model.generate 期望 'input' 作為第一個主要參數，'fs' 作為關鍵字參數
            func_to_run = partial(self.model.generate, input=audio_np, fs=sample_rate)

            logger.debug(f"{log_prefix} 調用模型 generate，輸入大小: {audio_np.shape}")
            model_output = await loop.run_in_executor(
                None,  # 使用默認的 ThreadPoolExecutor
                func_to_run  # 傳遞已綁定參數的函數
            )
            logger.debug(f"{log_prefix} 模型 generate 返回結果: {model_output}")
        except Exception as e:  # pragma: no cover
            logger.error(f"{log_prefix} 模型推理時發生錯誤: {e}", exc_info=True)
            raise ModuleProcessingError(f"FunASR [{self.module_id}] 模型推理錯誤: {e}") from e

        # 2. 解析模型輸出，只提取文本
        recognized_text = ""
        if model_output and isinstance(model_output, list) and len(model_output) > 0:
            texts = [item.get("text", "") for item in model_output if isinstance(item, dict) and "text" in item]
            recognized_text = " ".join(texts).strip()
            logger.debug(f"{log_prefix} 解析後識別文本: '{recognized_text}'")
        else:
            logger.debug(f"{log_prefix} FunASR 模型未返回有效的文本段。原始輸出: {model_output}")
            # recognized_text 保持為 ""

        return recognized_text if recognized_text.strip() else None
