import asyncio
import os
from functools import partial
from typing import Optional, Dict, Any, Type

import numpy as np

from src.core.models.exceptions import ModuleInitializationError, ModuleProcessingError
from src.core.models import AudioData
from src.core.interfaces.base_asr import BaseASR
from src.utils.audio_converter import convert_audio_format
from src.utils.logging_setup import logger

# 动态导入 FunASR
try:
    from funasr import AutoModel

    FUNASR_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning("funasr 库未安装，FunASRSenseVoiceAdapter 将无法使用。请运行 'pip install funasr'")
    AutoModel = None  # type: ignore
    FUNASR_AVAILABLE = False


class FunASRSenseVoiceAdapter(BaseASR):
    """FunASR SenseVoice 语音识别适配器

    使用 FunASR 的 SenseVoice 模型进行中文语音识别。
    """

    # 默认配置常量
    DEFAULT_MODEL_DIR = "models/asr/SenseVoiceSmall"
    DEFAULT_DEVICE = "cpu"
    DEFAULT_CHUNK_SIZE = [5, 10, 5]

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        if not FUNASR_AVAILABLE:
            raise ModuleInitializationError("funasr 库未安装")

        # 读取 FunASR 特定配置
        self.model_dir = self.config.get("model_dir", self.DEFAULT_MODEL_DIR)
        self.device = self.config.get("device", self.DEFAULT_DEVICE)
        self.vad_chunk_size = self.config.get("vad_chunk_size")
        self.output_dir = self.config.get("output_dir")

        self.model: Optional[AutoModel] = None

        logger.info(f"ASR/FunASR [{self.module_id}] 配置加载完成:")
        logger.info(f"  - 模型目录: {self.model_dir}")
        logger.info(f"  - 设备: {self.device}")

    async def _setup_impl(self):
        """初始化 FunASR SenseVoice 模型 (内部实现)"""
        logger.info(f"ASR/FunASR [{self.module_id}] 正在初始化模型...")

        try:
            # 验证模型路径
            model_path = os.path.abspath(self.model_dir)
            if not os.path.exists(model_path):
                raise ModuleInitializationError(f"模型目录不存在: {model_path}")

            logger.info(f"ASR/FunASR [{self.module_id}] 从 {model_path} 加载模型...")

            # 构建模型参数
            chunk_size = [
                self.vad_chunk_size or self.DEFAULT_CHUNK_SIZE[0],
                self.DEFAULT_CHUNK_SIZE[1],
                self.DEFAULT_CHUNK_SIZE[2],
            ]

            params = {
                "model": model_path,
                "device": self.device,
                "disable_pbar": True,
                "chunk_size": chunk_size,
            }

            if self.output_dir:
                params["output_dir"] = os.path.abspath(self.output_dir)

            logger.debug(f"ASR/FunASR [{self.module_id}] 模型参数: {params}")

            # 加载模型
            self.model = AutoModel(**params)  # type: ignore

            if self.model is None:
                raise ModuleInitializationError("AutoModel 返回 None")

            logger.info(f"ASR/FunASR [{self.module_id}] 模型初始化成功")

        except Exception as e:
            logger.error(f"ASR/FunASR [{self.module_id}] 初始化失败: {e}", exc_info=True)
            raise ModuleInitializationError(f"FunASR 初始化失败: {e}") from e

    async def recognize(self, audio: AudioData) -> str:
        """识别音频"""
        if self.model is None:
            raise ModuleProcessingError("模型未初始化")

        # 预处理音频
        audio_array = self._preprocess(audio)
        if audio_array is None or audio_array.size == 0:
            logger.debug(f"ASR/FunASR [{self.module_id}] 音频预处理后为空")
            return ""

        # 执行推理
        try:
            result = await self._infer(audio_array)

            # 提取文本
            text = self._extract_text(result)

            if text:
                logger.debug(f"ASR/FunASR [{self.module_id}] 识别结果: '{text}'")

            return text

        except Exception as e:
            logger.error(f"ASR/FunASR [{self.module_id}] 推理失败: {e}", exc_info=True)
            raise ModuleProcessingError(f"推理失败: {e}") from e

    def _preprocess(self, audio: AudioData) -> Optional[np.ndarray]:
        """预处理音频数据，将 AudioData 转换为模型所需的格式"""
        return convert_audio_format(
            audio=audio,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=2,
            output_format="pcm_f32le"
        )

    async def _infer(self, audio_array: np.ndarray) -> Any:
        """执行模型推理"""
        loop = asyncio.get_running_loop()

        logger.debug(f"ASR/FunASR [{self.module_id}] 开始推理，音频形状: {audio_array.shape}")

        result = await loop.run_in_executor(
            None,
            partial(self.model.generate, input=audio_array, fs=self.sample_rate)
        )

        logger.debug(f"ASR/FunASR [{self.module_id}] 推理完成，结果: {result}")

        return result

    async def _close_impl(self) -> None:
        """释放模型资源"""
        logger.info(f"ASR/FunASR [{self.module_id}] 正在关闭...")

        if self.model is not None:
            del self.model
            self.model = None

        # 清理可能存在的 CUDA 缓存（如果使用了 GPU）
        if self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"ASR/FunASR [{self.module_id}] 资源已释放")

    def _extract_text(self, result: Any) -> str:
        """从模型输出中提取文本"""
        if not result or not isinstance(result, list):
            return ""

        # 提取所有文本段
        text_segments = [
            item.get("text", "").strip()
            for item in result
            if isinstance(item, dict) and "text" in item
        ]

        return " ".join(text_segments).strip()


def load() -> Type["FunASRSenseVoiceAdapter"]:
    """加载 FunASRSenseVoice 适配器类"""
    return FunASRSenseVoiceAdapter
