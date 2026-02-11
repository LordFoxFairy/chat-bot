"""
端到端测试：文字 -> LLM -> TTS -> ASR 完整链路

测试流程:
1. 发送问题给 LLM: "用一句话介绍自己"
2. 获取 LLM 的文字回复
3. 将 LLM 回复通过 TTS 转换为语音
4. 将语音通过 ASR 识别回文字
5. 验证 ASR 识别结果与 LLM 原始回复相似（不需要完全一致）

运行方式:
    cd /Users/nako/PycharmProjects/github/thefoxfairy/chat-bot
    python -m pytest tests/e2e/test_llm_tts_asr_chain.py -v -s

    或直接运行:
    python tests/e2e/test_llm_tts_asr_chain.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from difflib import SequenceMatcher

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import yaml

from backend.core.models import AudioData, AudioFormat, TextData
from backend.utils.logging_setup import setup_logging, logger


def load_config() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = PROJECT_ROOT / "backend" / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（0-1）"""
    # 移除标点和空格进行比较
    import re
    clean1 = re.sub(r'[^\w]', '', text1)
    clean2 = re.sub(r'[^\w]', '', text2)
    return SequenceMatcher(None, clean1.lower(), clean2.lower()).ratio()


def contains_key_words(original: str, recognized: str) -> bool:
    """检查识别结果是否包含原文的关键词"""
    import re
    # 提取中文关键词（长度 >= 2 的连续中文字符）
    pattern = r'[\u4e00-\u9fa5]{2,}'
    original_words = set(re.findall(pattern, original))
    recognized_words = set(re.findall(pattern, recognized))

    if not original_words:
        return True  # 没有中文关键词时跳过检查

    # 计算关键词匹配率
    matched = original_words & recognized_words
    match_ratio = len(matched) / len(original_words)

    logger.info(f"关键词匹配: {matched}")
    logger.info(f"关键词匹配率: {match_ratio:.2%}")

    return match_ratio >= 0.3  # 至少 30% 的关键词匹配


async def create_llm_adapter(config: Dict[str, Any]):
    """创建并初始化 LLM 适配器"""
    from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter

    llm_config = config["modules"]["llm"]

    # 解析 enable_module 配置
    enable_module = llm_config.get("enable_module", "default")
    adapter_config = llm_config.get("config", {}).get(enable_module, {})

    # 合并顶层配置
    if "system_prompt" in llm_config:
        adapter_config["system_prompt"] = llm_config["system_prompt"]

    adapter = LangChainLLMAdapter(
        module_id="llm",
        config=adapter_config
    )
    await adapter.setup()
    return adapter


async def create_tts_adapter(config: Dict[str, Any]):
    """创建并初始化 TTS 适配器"""
    from backend.adapters.tts.edge_tts_adapter import EdgeTTSAdapter

    tts_config = config["modules"]["tts"]
    adapter_config = tts_config.get("config", {}).get("edge_tts", {})

    adapter = EdgeTTSAdapter(
        module_id="tts",
        config=adapter_config
    )
    await adapter.setup()
    return adapter


async def create_asr_adapter(config: Dict[str, Any]):
    """创建并初始化 ASR 适配器"""
    from backend.adapters.asr.funasr_sensevoice_adapter import FunASRSenseVoiceAdapter

    asr_config = config["modules"]["asr"]
    adapter_config = asr_config.get("config", {}).get("funasr_sensevoice", {})

    adapter = FunASRSenseVoiceAdapter(
        module_id="asr",
        config=adapter_config
    )
    await adapter.setup()
    return adapter


async def _run_llm_response(llm_adapter, question: str, session_id: str) -> str:
    """执行 LLM 回复步骤"""
    logger.info(f"\n{'='*60}")
    logger.info(f"步骤 1: 发送问题给 LLM")
    logger.info(f"问题: {question}")
    logger.info(f"{'='*60}")

    input_text = TextData(text=question)
    full_response = ""

    async for chunk in llm_adapter.chat_stream(input_text, session_id):
        if chunk.text:
            full_response += chunk.text
            print(chunk.text, end="", flush=True)

    print()  # 换行
    logger.info(f"\n步骤 2: LLM 回复完成")
    logger.info(f"完整回复: {full_response}")

    return full_response


async def _run_tts_synthesis(tts_adapter, text: str) -> bytes:
    """执行 TTS 语音合成步骤"""
    logger.info(f"\n{'='*60}")
    logger.info(f"步骤 3: 将文字转换为语音 (TTS)")
    logger.info(f"输入文本: {text}")
    logger.info(f"{'='*60}")

    input_text = TextData(text=text)
    audio_chunks = []

    async for audio_chunk in tts_adapter.synthesize_stream(input_text):
        if audio_chunk.data and len(audio_chunk.data) > 1:  # 排除占位符
            audio_chunks.append(audio_chunk.data)
            logger.debug(f"收到音频块: {len(audio_chunk.data)} 字节")

    # 合并所有音频块
    full_audio = b"".join(audio_chunks)
    logger.info(f"TTS 合成完成，总音频大小: {len(full_audio)} 字节")

    return full_audio


async def _run_asr_recognition(asr_adapter, audio_data: bytes) -> str:
    """执行 ASR 语音识别步骤"""
    logger.info(f"\n{'='*60}")
    logger.info(f"步骤 4: 将语音转换回文字 (ASR)")
    logger.info(f"输入音频大小: {len(audio_data)} 字节")
    logger.info(f"{'='*60}")

    # 创建 AudioData 对象（TTS 输出是 MP3 格式）
    audio = AudioData(
        data=audio_data,
        format=AudioFormat.MP3,
        sample_rate=16000,
        channels=1,
        sample_width=2
    )

    # 执行识别
    recognized_text = await asr_adapter.recognize(audio)

    logger.info(f"ASR 识别完成")
    logger.info(f"识别结果: {recognized_text}")

    return recognized_text


def clean_asr_output(text: str) -> str:
    """清理 ASR 输出中的特殊标记"""
    import re
    # 移除 SenseVoice 特有的标记，如 <|zh|>, <|NEUTRAL|>, <|Speech|>, <|woitn|> 等
    cleaned = re.sub(r'<\|[^>]+\|>', '', text)
    return cleaned.strip()


def verify_results(original_text: str, recognized_text: str) -> bool:
    """验证 ASR 识别结果与原始文本的相似度"""
    logger.info(f"\n{'='*60}")
    logger.info(f"步骤 5: 验证识别结果")
    logger.info(f"{'='*60}")
    logger.info(f"原始文本: {original_text}")
    logger.info(f"原始识别: {recognized_text}")

    # 清理 ASR 输出中的特殊标记
    cleaned_recognized = clean_asr_output(recognized_text)
    logger.info(f"清理后识别: {cleaned_recognized}")

    # 计算相似度
    similarity = calculate_similarity(original_text, cleaned_recognized)
    logger.info(f"文本相似度: {similarity:.2%}")

    # 检查关键词匹配
    keywords_match = contains_key_words(original_text, cleaned_recognized)
    logger.info(f"关键词匹配: {'通过' if keywords_match else '失败'}")

    # 验证条件：相似度 >= 30% 或 关键词匹配
    # 由于 ASR 可能有一些识别误差，我们使用较宽松的标准
    is_valid = similarity >= 0.3 or keywords_match

    logger.info(f"\n{'='*60}")
    logger.info(f"验证结果: {'通过' if is_valid else '失败'}")
    logger.info(f"{'='*60}")

    return is_valid, cleaned_recognized


async def run_e2e_test():
    """运行端到端测试"""
    print("\n" + "="*80)
    print("端到端测试：文字 -> LLM -> TTS -> ASR 完整链路")
    print("="*80 + "\n")

    # 加载配置
    config = load_config()

    # 设置日志
    setup_logging(config.get("logging", {}))

    llm_adapter = None
    tts_adapter = None
    asr_adapter = None

    try:
        # 创建适配器
        logger.info("正在初始化适配器...")

        llm_adapter = await create_llm_adapter(config)
        logger.info("LLM 适配器初始化完成")

        tts_adapter = await create_tts_adapter(config)
        logger.info("TTS 适配器初始化完成")

        asr_adapter = await create_asr_adapter(config)
        logger.info("ASR 适配器初始化完成")

        # 测试问题
        question = "用一句话介绍自己"
        session_id = "e2e_test_session"

        # 步骤 1-2: LLM 回复
        llm_response = await _run_llm_response(llm_adapter, question, session_id)

        if not llm_response or not llm_response.strip():
            logger.error("LLM 回复为空，测试失败")
            return False

        # 步骤 3: TTS 合成
        audio_data = await _run_tts_synthesis(tts_adapter, llm_response)

        if not audio_data or len(audio_data) < 100:
            logger.error("TTS 合成音频过小，测试失败")
            return False

        # 保存音频文件（用于调试）
        output_dir = PROJECT_ROOT / "outputs" / "e2e_test"
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_file = output_dir / "test_audio.mp3"
        with open(audio_file, "wb") as f:
            f.write(audio_data)
        logger.info(f"音频已保存到: {audio_file}")

        # 步骤 4: ASR 识别
        recognized_text = await _run_asr_recognition(asr_adapter, audio_data)

        if not recognized_text or not recognized_text.strip():
            logger.warning("ASR 识别结果为空")
            # 某些情况下 ASR 可能识别为空，但不一定是错误

        # 步骤 5: 验证结果
        is_valid, cleaned_recognized = verify_results(llm_response, recognized_text)

        # 输出测试摘要
        print("\n" + "="*80)
        print("测试摘要")
        print("="*80)
        print(f"问题: {question}")
        print(f"LLM 回复: {llm_response}")
        print(f"ASR 识别（清理后）: {cleaned_recognized}")
        print(f"测试结果: {'通过' if is_valid else '失败'}")
        print("="*80 + "\n")

        return is_valid

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        return False

    finally:
        # 清理资源
        logger.info("正在清理资源...")

        if llm_adapter:
            await llm_adapter.close()
        if tts_adapter:
            await tts_adapter.close()
        if asr_adapter:
            await asr_adapter.close()

        logger.info("资源清理完成")


# pytest 入口
async def test_e2e_llm_tts_asr():
    """pytest 测试入口"""
    result = await run_e2e_test()
    assert result, "端到端测试失败"


if __name__ == "__main__":
    # 直接运行测试
    success = asyncio.run(run_e2e_test())
    sys.exit(0 if success else 1)
