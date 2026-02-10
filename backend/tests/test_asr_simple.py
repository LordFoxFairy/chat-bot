"""简化ASR测试 - 直接使用FunASR"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 测试FunASR
from funasr import AutoModel

model_path = os.path.join(PROJECT_ROOT, "models/asr/SenseVoiceSmall/iic/SenseVoiceSmall")
example_file = os.path.join(model_path, "example/zh.mp3")

print(f"模型路径: {model_path}")
print(f"示例文件: {example_file}")
print(f"模型路径存在: {os.path.exists(model_path)}")
print(f"示例文件存在: {os.path.exists(example_file)}")

# 初始化模型
print("\n正在加载 FunASR 模型...")
model = AutoModel(
    model=model_path,
    device="cpu",
    disable_pbar=True,
)

print("✓ 模型加载成功")

# 识别音频
print(f"\n正在识别音频: {example_file}")
result = model.generate(
    input=example_file,
)

print(f"\n识别结果: {result}")

if result and isinstance(result, list) and len(result) > 0:
    text = result[0].get("text", "")
    print(f"\n✅ 识别成功: '{text}'")
else:
    print("\n❌ 识别失败")
