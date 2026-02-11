import asyncio
from typing import Dict, Any, Type

import numpy as np
import torch

from backend.core.models.exceptions import ModuleInitializationError, ModuleProcessingError
from backend.core.interfaces.base_vad import BaseVAD
from backend.utils.logging_setup import logger
from backend.utils.paths import resolve_project_path


class SileroVADAdapter(BaseVAD):
    """Silero VAD 语音活动检测适配器

    使用 Silero VAD 模型进行语音活动检测。
    """

    # 默认配置常量
    DEFAULT_MODEL_REPO = "outputs/models/vad/silero-vad"
    DEFAULT_MODEL_NAME = "silero_vad"
    DEFAULT_DEVICE = "cpu"
    DEFAULT_WINDOW_SIZE_16K = 512
    DEFAULT_WINDOW_SIZE_8K = 256

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(module_id, config)

        # 读取 Silero VAD 特定配置
        self.model_repo_path: str = self.config.get("model_repo_path", self.DEFAULT_MODEL_REPO)
        self.model_name: str = self.config.get("model_name", self.DEFAULT_MODEL_NAME)
        self.device: str = self.config.get("device", self.DEFAULT_DEVICE)
        self.force_reload: bool = self.config.get("force_reload_model", False)

        # 错误处理配置
        self.consecutive_failures: int = 0
        self.max_consecutive_failures: int = self.config.get("max_consecutive_failures", 10)

        # 根据采样率确定窗口大小
        default_window: int = (
            self.DEFAULT_WINDOW_SIZE_16K
            if self.sample_rate == 16000
            else self.DEFAULT_WINDOW_SIZE_8K
        )
        self.window_size_samples: int = self.config.get("window_size_samples", default_window)

        self.model: torch.nn.Module | None = None

        logger.info(f"VAD/Silero [{self.module_id}] 配置加载完成:")
        logger.info(f"  - model_repo: {self.model_repo_path}")
        logger.info(f"  - threshold: {self.threshold}")
        logger.info(f"  - sample_rate: {self.sample_rate}")
        logger.info(f"  - device: {self.device}")

    async def _setup_impl(self) -> None:
        """初始化 Silero VAD 模型 (内部实现)"""
        logger.info(f"VAD/Silero [{self.module_id}] 正在初始化模型...")

        try:
            # 判断是本地路径还是 GitHub
            repo_path = resolve_project_path(self.model_repo_path)
            is_local = repo_path.exists() and repo_path.is_dir()
            source_type = "local" if is_local else "github"

            # 如果是本地路径，使用绝对路径字符串
            model_source = str(repo_path) if is_local else self.model_repo_path

            logger.info(f"VAD/Silero [{self.module_id}] 从 {source_type} 加载模型: {model_source}")

            # 在线程池中加载模型
            loaded_entity = await asyncio.to_thread(
                torch.hub.load,
                repo_or_dir=model_source,
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

            logger.info(f"VAD/Silero [{self.module_id}] 模型初始化成功")

        except Exception as e:
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

            # Silero VAD 要求固定的 chunk size (512 for 16k, 256 for 8k)
            # 如果输入长度不是 512 的倍数，模型可能会报错
            # 这里我们只处理第一块符合要求的数据，或者进行 padding/truncate
            # 但更合理的做法是让上层保证传入的数据块大小正确，或者在这里进行缓冲
            # 根据错误信息，模型期望具体的 window size

            expected_size = self.window_size_samples
            if audio_tensor.shape[-1] != expected_size:
                # 简单处理：仅截取或填充以适应单个窗口大小，用于测试
                # 实际生产中应该有一个 buffer 机制
                if audio_tensor.shape[-1] > expected_size:
                    # 如果太长，只取第一帧
                    audio_tensor = audio_tensor[:expected_size]
                else:
                    # 如果太短，填充
                    padding = expected_size - audio_tensor.shape[-1]
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding), "constant", 0)

            # 执行检测
            with torch.no_grad():
                speech_prob_tensor = self.model(audio_tensor, self.sample_rate)
                speech_prob = speech_prob_tensor.item()

            is_speech = speech_prob >= self.threshold

            logger.debug(
                f"VAD/Silero [{self.module_id}] 检测结果: "
                f"is_speech={is_speech}, prob={speech_prob:.4f}, threshold={self.threshold}"
            )

            # 成功执行，重置失败计数
            self.consecutive_failures = 0

            return is_speech

        except Exception as e:
            self.consecutive_failures += 1
            logger.error(
                f"VAD/Silero [{self.module_id}] 检测失败 ({self.consecutive_failures}/{self.max_consecutive_failures}): {e}",
                exc_info=True
            )

            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.critical(f"VAD/Silero [{self.module_id}] 连续失败次数过多，抛出异常")
                raise ModuleProcessingError(f"Silero VAD 连续失败 {self.consecutive_failures} 次: {e}") from e

            return False

    async def reset_state(self) -> None:
        """重置 VAD 内部状态"""
        if self.model and hasattr(self.model, "reset_states"):
            self.model.reset_states()
            logger.debug(f"VAD/Silero [{self.module_id}] 模型状态已重置")
        await super().reset_state()

    async def _close_impl(self) -> None:
        """关闭模型，释放资源"""
        logger.info(f"VAD/Silero [{self.module_id}] 正在关闭...")

        if self.model:
            del self.model
            self.model = None

        # 清理 CUDA 缓存
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"VAD/Silero [{self.module_id}] 已关闭")


def load() -> Type["SileroVADAdapter"]:
    """加载 SileroVAD 适配器类"""
    return SileroVADAdapter
