# Chat-Bot 架构文档

## 项目概述

这是一个**基于 Python 和 WebSocket 的实时语音对话机器人框架**，采用高度模块化、可插拔的设计，支持低延迟、流式处理、对话打断和上下文感知等核心功能。

## 技术栈

- **语言**: Python 3.13+
- **核心依赖**:
  - WebSockets (通信协议)
  - LangChain (LLM 集成)
  - FunASR (语音识别)
  - Edge TTS (语音合成)
  - Silero VAD (语音活动检测)
  - Pydantic (数据验证)
  - PyTorch (深度学习模型)

## 目录结构

```
chat-bot/
├── adapters/              # 适配器层 - 具体实现与底层库集成
│   ├── asr/              # 语音识别适配器
│   ├── llm/              # 大语言模型适配器
│   ├── tts/              # 语音合成适配器
│   ├── vad/              # 语音活动检测适配器
│   └── protocols/        # 协议适配器
│
├── core/                  # 核心业务逻辑
│   ├── adapter_registry.py      # 通用适配器注册器
│   ├── app_context.py           # 全局应用上下文
│   ├── chat_engine.py           # 聊天引擎（总控制器）
│   ├── conversation_manager.py  # 会话管理器
│   ├── conversation.py          # 会话处理器
│   ├── session_manager.py       # 会话存储管理
│   ├── session_context.py       # 会话上下文
│   └── exceptions.py            # 自定义异常
│
├── modules/               # 抽象基类层 - 定义接口规范
│   ├── base_module.py    # 所有模块的基类
│   ├── base_protocol.py  # 协议模块基类
│   ├── base_asr.py       # ASR 模块基类
│   ├── base_llm.py       # LLM 模块基类
│   ├── base_tts.py       # TTS 模块基类
│   └── base_vad.py       # VAD 模块基类
│
├── handlers/              # 业务处理器
│   ├── audio_input.py    # 音频输入处理器
│   └── text_input.py     # 文本输入处理器
│
├── models/                # 数据模型
│   ├── stream_event.py   # 事件模型
│   ├── audio_data.py     # 音频数据模型
│   └── text_data.py      # 文本数据模型
│
├── utils/                 # 工具函数
├── configs/               # 配置文件
├── app.py                 # 应用启动入口
└── constant.py            # 常量定义
```

## 核心设计模式

### 1. 适配器注册器模式

所有适配器通过统一的 `AdapterRegistry` 管理，消除代码重复：

```python
from core.adapter_registry import AdapterRegistry
from modules.base_tts import BaseTTS

# 创建注册器
tts_registry = AdapterRegistry("TTS", BaseTTS)

# 注册适配器
tts_registry.register("edge_tts", "adapters.tts.edge_tts_adapter")

# 创建实例
tts = tts_registry.create("edge_tts", module_id="tts", config={})
```

### 2. 服务定位器模式 (AppContext)

全局模块通过 `AppContext` 访问，避免层层传递依赖：

```python
from core.app_context import AppContext
from modules.base_llm import BaseLLM

# 获取模块（带类型检查）
llm = AppContext.get_module_typed("llm", BaseLLM)
```

### 3. 分层架构

```
表示层 (Protocol) → 业务层 (Handlers/Conversation) → 服务层 (Modules/Adapters) → 数据层 (Models)
```

## 数据流

```
1. [客户端] 发送音频数据
   ↓
2. [WebSocketProtocol] 接收二进制数据
   ↓
3. [ConversationHandler] handle_audio()
   ↓
4. [AudioInputHandler] process_chunk()
   ↓
5. [VAD] detect() → 是否包含语音
   ↓
6. [ASR] recognize() → 文本
   ↓
7. [LLM] chat_stream() → 流式文本
   ↓
8. [TTS] synthesize_stream() → 流式音频
   ↓
9. [客户端] 播放音频
```

## 配置管理

### 安全配置

- 所有敏感配置（API Key 等）通过环境变量管理
- 配置文件模板：`configs/*.example`
- 实际配置文件已加入 `.gitignore`

### 环境变量

```bash
# .env 文件
DEEPSEEK_API_KEY=sk-your-key-here
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## 扩展指南

### 添加新的适配器

1. 创建适配器类，继承相应的基类
2. 在工厂中注册适配器
3. 更新配置文件

```python
# 1. 创建适配器
class MyTTSAdapter(BaseTTS):
    async def setup(self):
        ...

    async def synthesize_stream(self, text: TextData):
        ...

def load() -> Type["MyTTSAdapter"]:
    return MyTTSAdapter

# 2. 注册适配器
from adapters.tts import tts_registry
tts_registry.register("my_tts", "adapters.tts.my_tts_adapter")
```

## 最佳实践

1. **依赖注入**: 通过构造函数注入依赖，便于测试
2. **单向依赖**: 避免循环依赖，保持依赖方向清晰
3. **异步优先**: 所有 I/O 操作使用 async/await
4. **类型安全**: 使用类型注解，支持静态类型检查
5. **资源管理**: 实现 setup/close 生命周期方法
