#!/usr/bin/env python3
"""下载所需的模型文件"""
import os
import sys

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modelscope.hub.snapshot_download import snapshot_download


def download_sensevoice_model():
    """下载 FunASR SenseVoice 模型"""
    print("正在下载 SenseVoice 模型...")

    model_dir = os.path.join(PROJECT_ROOT, "models", "asr", "SenseVoiceSmall")

    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"模型已存在: {model_dir}")
        return model_dir

    try:
        cache_dir = snapshot_download(
            'iic/SenseVoiceSmall',
            cache_dir=model_dir,
        )
        print(f"✓ SenseVoice 模型下载成功: {cache_dir}")
        return cache_dir
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None


def download_silero_vad_model():
    """下载 Silero VAD 模型（通过 torch.hub）"""
    print("\n正在准备 Silero VAD 模型...")
    print("注意: Silero VAD 会在首次使用时自动下载")
    print("✓ 无需手动下载")


if __name__ == "__main__":
    print("=" * 60)
    print("开始下载模型")
    print("=" * 60)

    # 下载 SenseVoice
    download_sensevoice_model()

    # Silero VAD 提示
    download_silero_vad_model()

    print("\n" + "=" * 60)
    print("✅ 模型准备完成")
    print("=" * 60)
