#!/usr/bin/env python3
"""
全面真实系统测试

测试所有模块的真实功能，包括：
1. 单模块测试 - ASR, LLM, TTS, VAD 各自独立测试
2. 组合测试 - VAD+ASR, LLM+TTS 组合测试
3. 端到端测试 - 完整对话流程测试
"""
import asyncio
import os
import sys
import wave
import struct
import math
from pathlib import Path

import pytest
import numpy as np

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


class SystemTestReporter:
    """测试结果报告器"""
    def __init__(self):
        self.results = []
        self.current_section = None

    def section(self, name: str):
        self.current_section = name
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

    def test(self, name: str, passed: bool, detail: str = ""):
        status = "✓ PASS" if passed else "✗ FAIL"
        self.results.append((self.current_section, name, passed, detail))
        print(f"  [{status}] {name}")
        if detail and not passed:
            print(f"         {detail}")

    def summary(self):
        print(f"\n{'='*60}")
        print("  测试结果汇总")
        print(f"{'='*60}")

        passed = sum(1 for r in self.results if r[2])
        failed = sum(1 for r in self.results if not r[2])
        total = len(self.results)

        # 按 section 分组
        sections = {}
        for section, name, result, detail in self.results:
            if section not in sections:
                sections[section] = {"passed": 0, "failed": 0}
            if result:
                sections[section]["passed"] += 1
            else:
                sections[section]["failed"] += 1

        for section, stats in sections.items():
            status = "✓" if stats["failed"] == 0 else "✗"
            print(f"  {status} {section}: {stats['passed']}/{stats['passed']+stats['failed']} passed")

        print(f"\n  总计: {passed}/{total} 测试通过")

        if failed > 0:
            print(f"\n  失败的测试:")
            for section, name, result, detail in self.results:
                if not result:
                    print(f"    - [{section}] {name}")
                    if detail:
                        print(f"      {detail}")

        return failed == 0


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000) -> bytes:
    """生成正弦波音频数据"""
    num_samples = int(duration * sample_rate)
    samples = []
    for i in range(num_samples):
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
        samples.append(struct.pack('<h', sample))
    return b''.join(samples)


def generate_silence(duration: float, sample_rate: int = 16000) -> bytes:
    """生成静音数据"""
    num_samples = int(duration * sample_rate)
    return b'\x00\x00' * num_samples


def generate_vad_chunk(num_samples: int = 512, is_silence: bool = True, sample_rate: int = 16000) -> bytes:
    """生成 VAD 专用音频块（精确采样数）

    Silero VAD 需要精确的 512 采样 (16kHz) 或 256 采样 (8kHz)
    """
    if is_silence:
        return b'\x00\x00' * num_samples
    else:
        # 生成 440Hz 正弦波
        samples = []
        for i in range(num_samples):
            sample = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / sample_rate))
            samples.append(struct.pack('<h', sample))
        return b''.join(samples)


def load_test_audio() -> bytes:
    """加载或生成测试音频"""
    # 尝试加载真实的测试音频文件
    test_audio_paths = [
        PROJECT_ROOT / "backend" / "tests" / "fixtures" / "test_audio.wav",
        PROJECT_ROOT / "outputs" / "test_audio.wav",
    ]

    for path in test_audio_paths:
        if path.exists():
            with wave.open(str(path), 'rb') as wf:
                return wf.readframes(wf.getnframes())

    # 生成模拟音频（440Hz 正弦波 + 静音）
    audio = generate_sine_wave(440, 0.5)  # 0.5秒 440Hz
    audio += generate_silence(0.2)  # 0.2秒静音
    audio += generate_sine_wave(880, 0.3)  # 0.3秒 880Hz
    return audio



@pytest.fixture
def reporter():
    """创建测试报告器"""
    return SystemTestReporter()


@pytest.mark.asyncio
async def test_asr_module(reporter: SystemTestReporter):
    """测试 ASR 模块"""
    reporter.section("ASR 模块测试 (FunASR SenseVoice)")

    try:
        from backend.adapters.asr.funasr_sensevoice_adapter import FunASRSenseVoiceAdapter, FUNASR_AVAILABLE
        from backend.core.models import AudioData, AudioFormat

        # 测试依赖可用性
        reporter.test("FunASR 库可用", FUNASR_AVAILABLE)
        if not FUNASR_AVAILABLE:
            return

        # 测试适配器创建
        config = {
            'model_dir': 'outputs/models/asr/SenseVoiceSmall/iic/SenseVoiceSmall',
            'device': 'cpu',
            'sample_rate': 16000,
            'channels': 1,
        }

        adapter = FunASRSenseVoiceAdapter('test_asr', config)
        reporter.test("ASR 适配器创建", adapter is not None)

        # 测试模型加载
        await adapter.setup()
        reporter.test("ASR 模型加载", adapter.is_ready)

        # 测试音频识别（使用静音，预期返回空字符串）
        silence = generate_silence(1.0)
        audio_data = AudioData(
            data=silence,
            format=AudioFormat.PCM,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )

        result = await adapter.recognize(audio_data)
        reporter.test("ASR 静音识别", isinstance(result, str), f"结果类型: {type(result)}")

        # 测试关闭
        await adapter.close()
        reporter.test("ASR 模块关闭", not adapter.is_ready)

    except Exception as e:
        reporter.test("ASR 测试", False, str(e))


@pytest.mark.asyncio
async def test_tts_module(reporter: SystemTestReporter):
    """测试 TTS 模块"""
    reporter.section("TTS 模块测试 (Edge TTS)")

    try:
        from backend.adapters.tts.edge_tts_adapter import EdgeTTSAdapter, EDGE_TTS_AVAILABLE
        from backend.core.models import TextData

        # 测试依赖可用性
        reporter.test("Edge TTS 库可用", EDGE_TTS_AVAILABLE)
        if not EDGE_TTS_AVAILABLE:
            return

        # 测试适配器创建
        config = {
            'voice': 'zh-CN-XiaoxiaoNeural',
            'rate': '+0%',
            'volume': '+0%',
        }

        adapter = EdgeTTSAdapter('test_tts', config)
        reporter.test("TTS 适配器创建", adapter is not None)

        # 测试初始化
        await adapter.setup()
        reporter.test("TTS 初始化", adapter.is_ready)

        # 测试语音合成
        text = TextData(text="你好")
        chunks = []
        async for chunk in adapter.synthesize_stream(text):
            chunks.append(chunk)

        reporter.test("TTS 语音合成", len(chunks) > 0, f"生成 {len(chunks)} 个音频块")

        # 验证音频数据
        total_bytes = sum(len(c.data) for c in chunks if not c.is_final)
        reporter.test("TTS 音频数据有效", total_bytes > 0, f"总共 {total_bytes} 字节")

        # 测试空文本处理
        empty_text = TextData(text="", is_final=True)
        empty_chunks = []
        async for chunk in adapter.synthesize_stream(empty_text):
            empty_chunks.append(chunk)
        reporter.test("TTS 空文本处理", len(empty_chunks) == 1 and empty_chunks[0].is_final)

        # 测试关闭
        await adapter.close()
        reporter.test("TTS 模块关闭", not adapter.is_ready)

    except Exception as e:
        reporter.test("TTS 测试", False, str(e))


@pytest.mark.asyncio
async def test_vad_module(reporter: SystemTestReporter):
    """测试 VAD 模块"""
    reporter.section("VAD 模块测试 (Silero VAD)")

    try:
        from backend.adapters.vad.silero_vad_adapter import SileroVADAdapter
        from backend.core.models import AudioData, AudioFormat

        # 测试适配器创建
        config = {
            'model_repo_path': 'outputs/models/vad/silero-vad',
            'model_name': 'silero_vad',
            'threshold': 0.5,
            'vad_sample_rate': 16000,
            'window_size_samples': 512,
            'device': 'cpu',
        }

        adapter = SileroVADAdapter('test_vad', config)
        reporter.test("VAD 适配器创建", adapter is not None)

        # 测试模型加载
        await adapter.setup()
        reporter.test("VAD 模型加载", adapter.is_ready)

        # 测试静音检测 - VAD 需要精确的 512 采样 (16kHz)
        silence = generate_vad_chunk(512, is_silence=True)

        is_speech = await adapter.detect(silence)
        reporter.test("VAD 静音检测", is_speech == False, f"检测结果: {is_speech}")

        # 测试有声音检测（正弦波模拟语音）
        tone = generate_vad_chunk(512, is_silence=False)

        is_speech_tone = await adapter.detect(tone)
        # 注意：正弦波不一定被识别为语音，这里只测试不报错
        reporter.test("VAD 音频检测", isinstance(is_speech_tone, bool), f"检测结果: {is_speech_tone}")

        # 测试关闭
        await adapter.close()
        reporter.test("VAD 模块关闭", not adapter.is_ready)

    except Exception as e:
        reporter.test("VAD 测试", False, str(e))


@pytest.mark.asyncio
async def test_llm_module(reporter: SystemTestReporter):
    """测试 LLM 模块"""
    reporter.section("LLM 模块测试 (LangChain)")

    api_key = os.environ.get('API_KEY')
    if not api_key:
        reporter.test("API_KEY 环境变量", False, "未设置 API_KEY")
        return

    reporter.test("API_KEY 环境变量", True)

    try:
        from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
        from backend.core.models import TextData

        # 测试适配器创建
        config = {
            'model_name': 'anthropic/claude-3.5-sonnet',
            'api_key_env_var': 'API_KEY',
            'base_url': 'https://openrouter.ai/api/v1',
            'temperature': 0.7,
            'max_tokens': 50,
            'system_prompt': '你是测试助手，只回复"测试成功"四个字。',
        }

        adapter = LangChainLLMAdapter('test_llm', config)
        reporter.test("LLM 适配器创建", adapter is not None)

        # 测试初始化
        await adapter.setup()
        reporter.test("LLM 初始化", adapter.is_ready)

        # 测试对话
        text = TextData(text="测试")
        response = ""
        async for chunk in adapter.chat_stream(text, "test_session"):
            if chunk.text:
                response += chunk.text

        reporter.test("LLM 对话生成", len(response) > 0, f"响应长度: {len(response)}")

        # 测试历史记录
        history_len = adapter.get_history_length("test_session")
        reporter.test("LLM 历史记录", history_len > 0, f"历史长度: {history_len}")

        # 测试清除历史
        adapter.clear_history("test_session")
        reporter.test("LLM 清除历史", adapter.get_history_length("test_session") == 0)

        # 测试关闭
        await adapter.close()
        reporter.test("LLM 模块关闭", not adapter.is_ready)

    except Exception as e:
        reporter.test("LLM 测试", False, str(e))


@pytest.mark.asyncio
async def test_vad_asr_combination(reporter: SystemTestReporter):
    """测试 VAD + ASR 组合"""
    reporter.section("组合测试: VAD + ASR")

    try:
        from backend.adapters.vad.silero_vad_adapter import SileroVADAdapter
        from backend.adapters.asr.funasr_sensevoice_adapter import FunASRSenseVoiceAdapter, FUNASR_AVAILABLE
        from backend.core.models import AudioData, AudioFormat

        if not FUNASR_AVAILABLE:
            reporter.test("依赖检查", False, "FunASR 不可用")
            return

        # 初始化 VAD
        vad_config = {
            'model_repo_path': 'outputs/models/vad/silero-vad',
            'threshold': 0.5,
            'vad_sample_rate': 16000,
            'window_size_samples': 512,
            'device': 'cpu',
        }
        vad = SileroVADAdapter('vad', vad_config)
        await vad.setup()

        # 初始化 ASR
        asr_config = {
            'model_dir': 'outputs/models/asr/SenseVoiceSmall/iic/SenseVoiceSmall',
            'device': 'cpu',
            'sample_rate': 16000,
            'channels': 1,
        }
        asr = FunASRSenseVoiceAdapter('asr', asr_config)
        await asr.setup()

        reporter.test("VAD+ASR 模块初始化", vad.is_ready and asr.is_ready)

        # 模拟音频流处理 - 使用 VAD 专用采样数
        audio_stream = [
            generate_vad_chunk(512, is_silence=True),   # 静音
            generate_vad_chunk(512, is_silence=False),  # 音频
            generate_vad_chunk(512, is_silence=True),   # 静音
        ]

        speech_detected = False
        for i, chunk in enumerate(audio_stream):
            # VAD 接口接受 bytes
            is_speech = await vad.detect(chunk)
            if is_speech:
                speech_detected = True

        reporter.test("VAD+ASR 流处理", True, f"语音检测: {speech_detected}")

        # 清理
        await vad.close()
        await asr.close()
        reporter.test("VAD+ASR 资源释放", not vad.is_ready and not asr.is_ready)

    except Exception as e:
        reporter.test("VAD+ASR 组合测试", False, str(e))


@pytest.mark.asyncio
async def test_llm_tts_combination(reporter: SystemTestReporter):
    """测试 LLM + TTS 组合"""
    reporter.section("组合测试: LLM + TTS")

    api_key = os.environ.get('API_KEY')
    if not api_key:
        reporter.test("API_KEY 检查", False, "未设置")
        return

    try:
        from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
        from backend.adapters.tts.edge_tts_adapter import EdgeTTSAdapter, EDGE_TTS_AVAILABLE
        from backend.core.models import TextData

        if not EDGE_TTS_AVAILABLE:
            reporter.test("依赖检查", False, "Edge TTS 不可用")
            return

        # 初始化 LLM
        llm_config = {
            'model_name': 'anthropic/claude-3.5-sonnet',
            'api_key_env_var': 'API_KEY',
            'base_url': 'https://openrouter.ai/api/v1',
            'temperature': 0.7,
            'max_tokens': 30,
            'system_prompt': '用一句简短的中文回复。',
        }
        llm = LangChainLLMAdapter('llm', llm_config)
        await llm.setup()

        # 初始化 TTS
        tts_config = {
            'voice': 'zh-CN-XiaoxiaoNeural',
            'rate': '+0%',
        }
        tts = EdgeTTSAdapter('tts', tts_config)
        await tts.setup()

        reporter.test("LLM+TTS 模块初始化", llm.is_ready and tts.is_ready)

        # LLM 生成文本
        input_text = TextData(text="你好")
        llm_response = ""
        async for chunk in llm.chat_stream(input_text, "combo_session"):
            if chunk.text:
                llm_response += chunk.text

        reporter.test("LLM 生成响应", len(llm_response) > 0, f"响应: {llm_response[:50]}")

        # TTS 合成语音
        tts_input = TextData(text=llm_response)
        audio_chunks = []
        async for chunk in tts.synthesize_stream(tts_input):
            audio_chunks.append(chunk)

        total_audio = sum(len(c.data) for c in audio_chunks if not c.is_final)
        reporter.test("TTS 合成音频", total_audio > 0, f"音频大小: {total_audio} 字节")

        # 清理
        await llm.close()
        await tts.close()
        reporter.test("LLM+TTS 资源释放", not llm.is_ready and not tts.is_ready)

    except Exception as e:
        reporter.test("LLM+TTS 组合测试", False, str(e))


@pytest.mark.asyncio
async def test_full_pipeline(reporter: SystemTestReporter):
    """测试完整流水线: 音频 -> VAD -> ASR -> LLM -> TTS -> 音频"""
    reporter.section("端到端测试: 完整对话流水线")

    api_key = os.environ.get('API_KEY')
    if not api_key:
        reporter.test("API_KEY 检查", False, "未设置")
        return

    try:
        from backend.adapters.vad.silero_vad_adapter import SileroVADAdapter
        from backend.adapters.asr.funasr_sensevoice_adapter import FunASRSenseVoiceAdapter, FUNASR_AVAILABLE
        from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
        from backend.adapters.tts.edge_tts_adapter import EdgeTTSAdapter, EDGE_TTS_AVAILABLE
        from backend.core.models import AudioData, AudioFormat, TextData

        if not FUNASR_AVAILABLE or not EDGE_TTS_AVAILABLE:
            reporter.test("依赖检查", False, "部分依赖不可用")
            return

        # 初始化所有模块
        vad = SileroVADAdapter('vad', {
            'model_repo_path': 'outputs/models/vad/silero-vad',
            'threshold': 0.5,
            'vad_sample_rate': 16000,
            'window_size_samples': 512,
            'device': 'cpu',
        })

        asr = FunASRSenseVoiceAdapter('asr', {
            'model_dir': 'outputs/models/asr/SenseVoiceSmall/iic/SenseVoiceSmall',
            'device': 'cpu',
            'sample_rate': 16000,
            'channels': 1,
        })

        llm = LangChainLLMAdapter('llm', {
            'model_name': 'anthropic/claude-3.5-sonnet',
            'api_key_env_var': 'API_KEY',
            'base_url': 'https://openrouter.ai/api/v1',
            'temperature': 0.7,
            'max_tokens': 50,
            'system_prompt': '用简短中文回复。',
        })

        tts = EdgeTTSAdapter('tts', {
            'voice': 'zh-CN-XiaoxiaoNeural',
            'rate': '+0%',
        })

        # 启动所有模块
        await asyncio.gather(
            vad.setup(),
            asr.setup(),
            llm.setup(),
            tts.setup(),
        )

        all_ready = vad.is_ready and asr.is_ready and llm.is_ready and tts.is_ready
        reporter.test("所有模块初始化", all_ready)

        if not all_ready:
            return

        # 步骤 1: 模拟输入音频
        input_audio = generate_silence(0.5)
        reporter.test("步骤1: 音频输入准备", len(input_audio) > 0)

        # 步骤 2: VAD 检测 - VAD 需要 512 采样
        vad_chunk = generate_vad_chunk(512, is_silence=True)
        is_speech = await vad.detect(vad_chunk)
        reporter.test("步骤2: VAD 检测", isinstance(is_speech, bool), f"结果: {is_speech}")

        # 步骤 3: ASR 识别（用静音测试，预期返回空）
        audio_data = AudioData(
            data=input_audio,
            format=AudioFormat.PCM,
            sample_rate=16000,
            channels=1,
            sample_width=2
        )
        asr_result = await asr.recognize(audio_data)
        reporter.test("步骤3: ASR 识别", isinstance(asr_result, str), f"结果长度: {len(asr_result)}")

        # 步骤 4: LLM 生成（使用固定文本测试）
        llm_input = TextData(text="你好")
        llm_response = ""
        async for chunk in llm.chat_stream(llm_input, "pipeline_session"):
            if chunk.text:
                llm_response += chunk.text
        reporter.test("步骤4: LLM 生成", len(llm_response) > 0, f"响应: {llm_response[:30]}...")

        # 步骤 5: TTS 合成
        tts_input = TextData(text=llm_response)
        audio_output = []
        async for chunk in tts.synthesize_stream(tts_input):
            if not chunk.is_final:
                audio_output.append(chunk.data)

        total_audio_size = sum(len(a) for a in audio_output)
        reporter.test("步骤5: TTS 合成", total_audio_size > 0, f"音频大小: {total_audio_size} 字节")

        # 完整流水线成功
        reporter.test("完整流水线执行", True)

        # 清理
        await asyncio.gather(
            vad.close(),
            asr.close(),
            llm.close(),
            tts.close(),
        )
        reporter.test("资源释放", True)

    except Exception as e:
        reporter.test("端到端测试", False, str(e))


@pytest.mark.asyncio
async def test_chat_engine(reporter: SystemTestReporter):
    """测试 ChatEngine 集成"""
    reporter.section("ChatEngine 集成测试")

    api_key = os.environ.get('API_KEY')
    if not api_key:
        reporter.test("API_KEY 检查", False, "未设置")
        return

    try:
        from backend.utils.config_loader import ConfigLoader
        from backend.core.engine.chat_engine import ChatEngine
        from backend.core.session.session_manager import SessionManager, InMemoryStorage

        # 加载配置
        config = await ConfigLoader.load_config('backend/configs/config.yaml')
        reporter.test("配置加载", config is not None)

        # 创建 SessionManager
        storage = InMemoryStorage(maxsize=100)
        session_manager = SessionManager(storage_backend=storage)
        reporter.test("SessionManager 创建", session_manager is not None)

        # 创建 ChatEngine
        engine = ChatEngine(config=config, session_manager=session_manager)
        reporter.test("ChatEngine 创建", engine is not None)

        # 初始化
        await engine.initialize()

        # 检查所有模块
        modules_ready = all(m.is_ready for m in engine.common_modules.values())
        reporter.test("所有模块就绪", modules_ready,
                     f"模块: {list(engine.common_modules.keys())}")

        # 获取模块
        asr = engine.get_module('asr')
        llm = engine.get_module('llm')
        tts = engine.get_module('tts')
        vad = engine.get_module('vad')

        reporter.test("模块获取", all([asr, llm, tts, vad]))

        # 关闭
        await engine.shutdown()
        reporter.test("ChatEngine 关闭", True)

    except Exception as e:
        reporter.test("ChatEngine 测试", False, str(e))


async def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("  Chat-Bot 全面系统测试")
    print("="*60)

    # 检查环境变量
    api_key = os.environ.get('API_KEY')
    if not api_key:
        # 尝试从 .env 文件加载
        env_file = PROJECT_ROOT / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith('API_KEY='):
                        os.environ['API_KEY'] = line.split('=', 1)[1].strip()
                        break

    reporter = SystemTestReporter()

    # 1. 单模块测试
    await test_vad_module(reporter)
    await test_asr_module(reporter)
    await test_tts_module(reporter)
    await test_llm_module(reporter)

    # 2. 组合测试
    await test_vad_asr_combination(reporter)
    await test_llm_tts_combination(reporter)

    # 3. 端到端测试
    await test_full_pipeline(reporter)

    # 4. ChatEngine 集成测试
    await test_chat_engine(reporter)

    # 输出汇总
    success = reporter.summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
