#!/usr/bin/env python3
"""下载 Silero VAD 模型到本地 .cache/models 目录"""
import os
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # backend/scripts -> backend -> project_root


def get_cache_dir() -> Path:
    """获取模型缓存目录"""
    cache_dir = os.environ.get("CHATBOT_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "models"
    return PROJECT_ROOT / ".cache" / "models"


def download_silero_vad() -> Path | None:
    """下载 Silero VAD 模型"""
    print("正在下载 Silero VAD 模型...")

    model_dir = get_cache_dir() / "vad" / "silero-vad"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已存在
    if (model_dir / "silero_vad.jit").exists() or (model_dir / "model.py").exists():
        print(f"模型已存在: {model_dir}")
        return model_dir

    try:
        import torch

        # 使用 torch.hub 下载模型
        print("正在通过 torch.hub 下载...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
        )

        # 保存模型状态
        model_path = model_dir / "silero_vad.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✓ 模型状态已保存: {model_path}")

        # 克隆完整仓库以获取所有文件
        print("\n克隆 Silero VAD 仓库...")
        repo_url = "https://github.com/snakers4/silero-vad.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(model_dir)],
            check=False
        )

        print(f"\n✅ Silero VAD 模型准备完成: {model_dir}")
        return model_dir

    except ImportError:
        print("✗ 缺少 torch 依赖，请先安装: uv pip install torch")
        return None
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n备选方案: 手动下载")
        print("1. 访问: https://github.com/snakers4/silero-vad")
        print(f"2. 克隆到: {model_dir}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Silero VAD 模型下载工具")
    print(f"目标目录: {get_cache_dir() / 'vad' / 'silero-vad'}")
    print("=" * 60)

    download_silero_vad()
