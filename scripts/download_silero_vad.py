#!/usr/bin/env python3
"""下载 Silero VAD 模型到本地"""
import os
import sys
import torch

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def download_silero_vad():
    """下载 Silero VAD 模型"""
    print("正在下载 Silero VAD 模型...")

    model_dir = os.path.join(PROJECT_ROOT, "models", "vad", "silero_vad")
    os.makedirs(model_dir, exist_ok=True)

    try:
        # 使用 torch.hub 下载模型
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=True,
        )

        # 保存模型到本地
        model_path = os.path.join(model_dir, "silero_vad.pt")
        torch.save(model.state_dict(), model_path)

        print(f"✓ Silero VAD 模型已下载到: {model_dir}")
        print(f"  - 模型文件: {model_path}")

        # 克隆 silero-vad 仓库以获取完整文件
        print("\n下载 Silero VAD 仓库...")
        import subprocess
        repo_url = "https://github.com/snakers4/silero-vad.git"
        subprocess.run(["git", "clone", repo_url, model_dir], check=False)

        print("\n✅ Silero VAD 模型准备完成")
        print(f"模型目录: {model_dir}")

        return model_dir

    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n备选方案: 手动下载")
        print("1. 访问: https://github.com/snakers4/silero-vad")
        print(f"2. 克隆到: {model_dir}")
        return None


if __name__ == "__main__":
    download_silero_vad()
