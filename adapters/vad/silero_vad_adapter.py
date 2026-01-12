import asyncio
import os
from typing import Optional, Dict, Any

import numpy as np
import torch

from core.exceptions import ModuleInitializationError, ModuleProcessingError
from modules.base_vad import BaseVAD
from utils.logging_setup import logger

# 动态导入 torch
try:
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning("torch 库未安装，SileroVADAdapter 将无法使用。请运行 'pip install torch'")
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


class SileroVADAdapter(BaseVAD):
    """Silero VAD 语音活动检测适配器

    使用 Silero VAD 模型进行语音活动检测。
    """

    # 默认配置常量
    DEFAULT_MODEL_REPO = "snakers4/silero-vad"
    DEFAULT_MODEL_NAME = "silero_vad"
    DEFAULT_DEVICE = "cpu"
    DEFAULT_WINDOW_SIZE_16K = 512
    DEFAULT_WINDOW_SIZE_8K = 256

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        if not TORCH_AVAILABLE:
            raise ModuleInitializationError("torch 库未安装")

        # 读取 Silero VAD 特定配置
        self.model_repo_path = self.config.get("model_repo_path", self.DEFAULT_MODEL_REPO)
        self.model_name = self.config.get("model_name", self.DEFAULT_MODEL_NAME)
        self.device = self.config.get("device", self.DEFAULT_DEVICE)
        self.force_reload = self.config.get("force_reload_model", False)

        # 根据采样率确定窗口大小
        default_window = (
            self.DEFAULT_WINDOW_SIZE_16K
            if self.sample_rate == 16000
            else self.DEFAULT_WINDOW_SIZE_8K
        )
        self.window_size_samples = self.config.get("window_size_samples", default_window)

        self.model: Optional[torch.nn.Module] = None

        logger.info(f"VAD/Silero [{self.module_id}] 配置加载完成:")
        logger.info(f"  - model_repo: {self.model_repo_path}")
        logger.info(f"  - threshold: {self.threshold}")
        logger.info(f"  - sample_rate: {self.sample_rate}")
        logger.info(f"  - device: {self.device}")

    async def setup(self):
        """初始化 Silero VAD 模型"""
        logger.info(f"VAD/Silero [{self.module_id}] 正在初始化模型...")

        try:
            # 判断是本地路径还是 GitHub
            is_local = os.path.exists(self.model_repo_path) and os.path.isdir(self.model_repo_path)
            source_type = "local" if is_local else "github"

            logger.info(f"VAD/Silero [{self.module_id}] 从 {source_type} 加载模型: {self.model_repo_path}")

            # 在线程池中加载模型
            loaded_entity = await asyncio.to_thread(
                torch.hub.load,
                repo_or_dir=self.model_repo_path,
                model=self.model_name,
                source=source_type,
                force_reload=self.force_reload,
                trust_repo=True if is_local else None
            )

            # 处理返回值（可能是 tuple）
            if isinstance(loaded_entity, tuple) and len(loaded_entity) >= 1:
                self.model = loaded_entity[0]
            else:
                self.model = loaded_entity

            if self.model is None:
                raise ModuleInitializationError("torch.hub.load 返回 None")

            # 设置设备和评估模式
            self.model.to(self.device)
            self.model.eval()

            self._is_initialized = True
            self._is_ready = True
            logger.info(f"VAD/Silero [{self.module_id}] 模型初始化成功")

        except Exception as e:
            self._is_initialized = False
            self._is_ready = False
            logger.error(f"VAD/Silero [{self.module_id}] 初始化失败: {e}", exc_info=True)
            raise ModuleInitializationError(f"Silero VAD 初始化失败: {e}") from e

    async def detect(self, audio_data: bytes) -> bool:
        """检测音频中是否包含语音"""
        if not self.model:
            raise ModuleProcessingError("模型未初始化")

        if not audio_data:
            logger.debug(f"VAD/Silero [{self.module_id}] 音频数据为空")
            return False

        try:
            # 转换音频数据
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float32).to(self.device)

            # 处理维度
            if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.squeeze(0)
            elif audio_tensor.ndim != 1:
                logger.error(f"VAD/Silero [{self.module_id}] 音频张量形状错误: {audio_tensor.shape}")
                return False

            # 执行检测
            with torch.no_grad():
                speech_prob_tensor = self.model(audio_tensor, self.sample_rate)
                speech_prob = speech_prob_tensor.item()

            is_speech = speech_prob >= self.threshold

            logger.debug(
                f"VAD/Silero [{self.module_id}] 检测结果: "
                f"is_speech={is_speech}, prob={speech_prob:.4f}, threshold={self.threshold}"
            )

            return is_speech

        except Exception as e:
            logger.error(f"VAD/Silero [{self.module_id}] 检测失败: {e}", exc_info=True)
            return False

    async def reset_state(self):
        """重置 VAD 内部状态"""
        if self.model and hasattr(self.model, "reset_states"):
            self.model.reset_states()
            logger.debug(f"VAD/Silero [{self.module_id}] 模型状态已重置")
        await super().reset_state()

    async def close(self):
        """关闭模型，释放资源"""
        logger.info(f"VAD/Silero [{self.module_id}] 正在关闭...")

        if self.model:
            del self.model
            self.model = None

        # 清理 CUDA 缓存
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._is_ready = False
        self._is_initialized = False

        logger.info(f"VAD/Silero [{self.module_id}] 已关闭")
        await super().close()
