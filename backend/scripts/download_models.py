#\!/usr/bin/env python3
"""下载所需的 AI 模型文件到 outputs/models 目录"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# 计算项目根目录: backend/scripts -> backend -> project_root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def get_cache_dir() -> Path:
    """获取模型缓存目录"""
    cache_dir = os.environ.get("CHATBOT_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir)
    return PROJECT_ROOT / "outputs" / "models"


def download_sensevoice_model() -> Path | None:
    """下载 FunASR SenseVoice 模型"""
    print("正在下载 SenseVoice 模型...")

    model_dir = get_cache_dir() / "asr" / "SenseVoiceSmall"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 检查模型是否已存在
    inner_model_dir = model_dir / "iic" / "SenseVoiceSmall"
    if inner_model_dir.exists() and any(inner_model_dir.iterdir()):
        print(f"✓ SenseVoice 模型已存在: {inner_model_dir}")
        return inner_model_dir

    try:
        from modelscope.hub.snapshot_download import snapshot_download

        cache_dir = snapshot_download(
            "iic/SenseVoiceSmall",
            cache_dir=str(model_dir),
        )
        print(f"✓ SenseVoice 模型下载成功: {cache_dir}")
        return Path(cache_dir)
    except ImportError:
        print("✗ 缺少 modelscope 依赖，请先安装: uv pip install modelscope")
        return None
    except Exception as e:
        print(f"✗ SenseVoice 下载失败: {e}")
        return None


def download_silero_vad_model() -> None:
    """下载 Silero VAD 模型仓库"""
    print("\\n正在处理 Silero VAD 模型...")
    target_dir = get_cache_dir() / "vad" / "silero-vad"

    # 检查是否已存在
    if target_dir.exists() and (target_dir / "hubconf.py").exists():
        print(f"✓ Silero VAD 模型已存在: {target_dir}")
        return

    print(f"正在克隆 Silero VAD 仓库到 {target_dir}...")
    try:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)

        subprocess.run(
            ["git", "clone", "https://github.com/snakers4/silero-vad", str(target_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✓ Silero VAD 下载成功")
    except subprocess.CalledProcessError as e:
        print(f"✗ Silero VAD 下载失败 (git clone): {e}")
    except FileNotFoundError:
         print("✗ 未找到 git 命令，无法克隆 Silero VAD 仓库")
    except Exception as e:
        print(f"✗ Silero VAD 下载失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Chat-Bot 模型下载工具")
    print(f"缓存目录: {get_cache_dir()}")
    print("=" * 60)

    # 下载 SenseVoice ASR 模型
    download_sensevoice_model()

    # 下载 Silero VAD 模型
    download_silero_vad_model()

    print("\\n" + "=" * 60)
    print("✅ 模型准备完成检查")
    print("=" * 60)
