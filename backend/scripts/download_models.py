#!/usr/bin/env python3
"""下载所需的 AI 模型文件到 .cache/models 目录"""
import os
from pathlib import Path

# 计算项目根目录: backend/scripts -> backend -> project_root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def get_cache_dir() -> Path:
    """获取模型缓存目录"""
    cache_dir = os.environ.get("CHATBOT_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "models"
    return PROJECT_ROOT / ".cache" / "models"


def download_sensevoice_model() -> Path | None:
    """下载 FunASR SenseVoice 模型"""
    print("正在下载 SenseVoice 模型...")

    model_dir = get_cache_dir() / "asr" / "SenseVoiceSmall"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 检查模型是否已存在
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"模型已存在: {model_dir}")
        return model_dir

    try:
        from modelscope.hub.snapshot_download import snapshot_download

        cache_dir = snapshot_download(
            'iic/SenseVoiceSmall',
            cache_dir=str(model_dir),
        )
        print(f"✓ SenseVoice 模型下载成功: {cache_dir}")
        return Path(cache_dir)
    except ImportError:
        print("✗ 缺少 modelscope 依赖，请先安装: uv pip install modelscope")
        return None
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None


def download_silero_vad_model() -> None:
    """提示 Silero VAD 模型信息"""
    print("\n正在准备 Silero VAD 模型...")
    print("注意: Silero VAD 会在首次使用时通过 torch.hub 自动下载")
    print("✓ 无需手动下载")


if __name__ == "__main__":
    print("=" * 60)
    print("Chat-Bot 模型下载工具")
    print(f"缓存目录: {get_cache_dir()}")
    print("=" * 60)

    # 下载 SenseVoice ASR 模型
    download_sensevoice_model()

    # Silero VAD 提示
    download_silero_vad_model()

    print("\n" + "=" * 60)
    print("✅ 模型准备完成")
    print("=" * 60)
