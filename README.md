## 整体架构

client <- ws -> server || data[audio] <- vad ->  asr
                            |                           <- llm/stream ->  text   <- stream ->  tts  <---> data[text/audio] 
                          data[text]

1. 客户端与服务端通过 ws 通信
2. 输入音频 - 【经过 var asr 转换为文本】可选
3. 输入文本
4. 给到大模型 - 得到回复 - 根据回复执行
5. 目前是 回复 -> 文本 -> 如果启动tts，则转换为 音频
6. 将文本和音频一起发给 client


## 目录结构

```markdown
conversational_ai_framework/
├── app.py                          # 主应用入口 (示例，例如启动FastAPI服务)
├── configs/                        # 配置文件目录
│   ├── default_pipeline.yaml
│   └── robot_control_pipeline.yaml
│
├── core_framework/                 # 核心框架层
│   ├── __init__.py
│   ├── pipeline_manager.py         # 流水线管理器
│   ├── module_manager.py           # 模块管理器
│   ├── session_manager.py          # 会话管理器
│   ├── event_manager.py            # 事件/信号总线
│   └── exceptions.py               # 框架自定义异常
│
├── data_models/                    # 核心数据结构定义 (基础设施层的一部分)
│   ├── __init__.py
│   ├── stream_event.py             # StreamEvent, ControlSignal, SignalType
│   ├── audio_data.py
│   ├── text_data.py
│   ├── motion_command_data.py      # MotionCommandData, MotionType
│   └── motion_feedback_data.py     # MotionFeedbackData, MotionStatus
│
├── modules/                        # 模块接口层 (抽象基类)
│   ├── __init__.py
│   ├── base_module.py              # BasePipelineModule
│   ├── base_asr.py
│   ├── base_llm.py
│   ├── base_tts.py
│   ├── base_vad.py
│   ├── base_tool_executor.py
│   └── base_robot_controller.py
│
├── adapters/                       # 模块适配器层 (具体实现)
│   ├── __init__.py
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── whisper_adapter.py
│   │   └── dummy_asr_adapter.py    # 用于测试的伪实现
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── openai_adapter.py
│   │   └── dummy_llm_adapter.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── edge_tts_adapter.py
│   │   └── dummy_tts_adapter.py
│   ├── vad/
│   │   └── ...
│   ├── tool_executors/
│   │   ├── __init__.py
│   │   ├── weather_tool_adapter.py
│   │   └── calculator_tool_adapter.py
│   └── robot_controllers/
│       ├── __init__.py
│       ├── ros_controller_adapter.py
│       └── dummy_robot_controller_adapter.py
│
├── services/                       # 基础设施与服务层 (除数据模型外)
│   ├── __init__.py
│   ├── config_loader.py            # 配置加载服务
│   ├── logging_setup.py            # 日志配置
│   └── communication/              # (可选) 底层通信协议封装
│       ├── __init__.py
│       ├── webrtc_handler.py
│       └── websocket_handler.py
│
├── applications/                   # 应用层示例 (可选，也可作为独立项目)
│   ├── __init__.py
│   ├── cli_chat_app.py             # 命令行聊天应用
│   └── web_voice_assistant/        # Web语音助手示例 (FastAPI + 前端)
│
├── tests/                          # 单元测试和集成测试
│   ├── core_framework/
│   ├── modules/
│   └── adapters/
│
├── utils/                          # 通用工具函数 (基础设施层的一部分)
│   ├── __init__.py
│   ├── async_utils.py
│   └── data_conversion.py
│
├── README.md
├── requirements.txt
└── .env.example                    # 环境变量示例

```