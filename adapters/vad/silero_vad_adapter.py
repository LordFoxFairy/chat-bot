import asyncio
import os
from typing import Optional, Dict, Any

import numpy as np
import torch

from modules.base_vad import BaseVAD
from utils.logging_setup import logger


class SileroVADAdapter(BaseVAD):
    """
    使用 Silero VAD 模型的 VAD 适配器。
    实现 is_speech_present 方法，对传入的单个音频窗口进行语音活动判断，返回布尔值。
    """

    def __init__(self, module_id: str,
                 config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None):
        super().__init__(
            module_id=module_id,
            config=config,
        )

        adapter_specific_parent_config = self.config.get('config', {})
        specific_silero_config = {}
        if isinstance(adapter_specific_parent_config, dict):
            specific_silero_config = adapter_specific_parent_config.get(self.get_config_key(), {})

        if not specific_silero_config:
            logger.warning(
                f"SileroVADAdapter [{self.module_id}]: 在 'modules.vad.config.{self.get_config_key()}' 中未找到特定配置，将尝试从 'modules.vad' 顶层读取。")
            specific_silero_config = self.config

        self.model_repo_path: str = specific_silero_config.get('model_repo_path', 'snakers4/silero-vad')
        self.model_name: str = specific_silero_config.get('model_name', 'silero_vad')
        self.threshold: float = float(specific_silero_config.get('threshold', 0.5))
        self.vad_sample_rate: int = int(specific_silero_config.get('vad_sample_rate', self.default_sample_rate))
        self.device: str = specific_silero_config.get('device', 'cpu')
        self.force_reload_model: bool = bool(specific_silero_config.get('force_reload_model', False))
        self.trust_repo_for_local: bool = bool(specific_silero_config.get('trust_repo_for_local', True))

        self.window_size_samples: int = int(
            specific_silero_config.get('window_size_samples', 512 if self.vad_sample_rate == 16000 else 256))
        self.model = None

        logger.info(f"SileroVADAdapter (Boolean Interface) [{self.module_id}] __init__ 完成。")
        logger.info(
            f"  VAD 参数: SR={self.vad_sample_rate}, Threshold={self.threshold}, Expected WindowSamples={self.window_size_samples}")

    async def initialize(self):
        logger.info(f"SileroVADAdapter [{self.module_id}] 开始异步初始化 (Boolean Interface)...")
        try:
            is_local_path = os.path.exists(self.model_repo_path) and os.path.isdir(self.model_repo_path)
            source_type = 'local' if is_local_path else 'github'

            loaded_entity = await asyncio.to_thread(
                torch.hub.load,
                repo_or_dir=self.model_repo_path, model=self.model_name,
                source=source_type, force_reload=self.force_reload_model,
                trust_repo=self.trust_repo_for_local if is_local_path else None
            )
            if isinstance(loaded_entity, tuple) and len(loaded_entity) >= 1:
                self.model = loaded_entity[0]
            else:
                self.model = loaded_entity

            if self.model is None:
                raise RuntimeError("torch.hub.load 未能按预期返回模型对象。")

            self.model.to(self.device)
            self.model.eval()

            logger.info(
                f"Silero VAD 模型已从 {source_type} 源 '{self.model_repo_path}' 成功加载到设备 '{self.device}' (Boolean Interface)。")
            self._is_initialized = True
            self._is_ready = True
        except Exception as e:
            logger.error(f"加载 Silero VAD 模型失败 (Boolean Interface): {e}", exc_info=True)
            self._is_initialized = False
            self._is_ready = False
            raise RuntimeError(f"Silero VAD 模型加载失败 (Boolean Interface): {e}") from e
        logger.info(f"SileroVADAdapter [{self.module_id}] 异步初始化完成 (Boolean Interface)。")

    def get_config_key(self) -> str:
        return "silero_vad"

    async def reset_state(self):
        if self.model and hasattr(self.model, 'reset_states') and callable(self.model.reset_states):
            self.model.reset_states()
            logger.debug(f"SileroVADAdapter [{self.module_id}] (Boolean Interface) model states reset.")
        else:
            logger.debug(
                f"SileroVADAdapter [{self.module_id}] (Boolean Interface) reset_state called, but model has no reset_states method or model not loaded.")
        await super().reset_state()

    async def is_speech_present(self, audio_data: bytes) -> bool:
        """
        对传入的单个音频窗口进行语音活动判断，返回 True 或 False。
        """
        log_prefix = f"SileroVADAdapter [{self.module_id}] (is_speech_present)"

        if not self.is_ready or not self.model:
            logger.error(f"{log_prefix} VAD 模型未就绪。")
            return False

        if not audio_data:
            logger.debug(f"{log_prefix}: 收到空音频数据。")
            return False  # 空数据认为无语音

        try:
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32).to(self.device)

            if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            elif audio_tensor.ndim != 1:
                logger.error(f"{log_prefix} 音频张量形状不正确: {audio_tensor.shape}")
                return False

            with torch.no_grad():
                speech_prob_tensor = self.model(audio_tensor, self.vad_sample_rate)
                speech_prob = speech_prob_tensor.item()

            is_speech_in_chunk = speech_prob >= self.threshold
            logger.debug(
                f"{log_prefix} VAD 判断: is_speech={is_speech_in_chunk}, "
                f"prob={speech_prob: .4f} (阈值={self.threshold})")
            return is_speech_in_chunk

        except Exception as e:
            logger.error(f"{log_prefix} VAD判断时发生错误: {e}", exc_info=True)
            return False  # 出错时返回 False

    async def close(self):
        await super().close()
        logger.info(f"正在关闭 SileroVADAdapter (Boolean Interface) [{self.module_id}]...")
        if hasattr(self, 'model') and self.model: del self.model
        self.model = None
        if self.device == 'cuda' and torch.cuda.is_available() and hasattr(torch.cuda,
                                                                           'empty_cache'):  # pragma: no cover
            torch.cuda.empty_cache()
        self._is_ready = False
        self._is_initialized = False
        logger.info(f"SileroVADAdapter (Boolean Interface) [{self.module_id}] 已关闭。")
