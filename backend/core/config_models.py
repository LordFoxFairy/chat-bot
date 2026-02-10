"""配置模型定义

提供 Pydantic 配置模型，用于配置验证和类型安全。
"""

from typing import Dict, Any, Optional, Literal, List
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ==================== 日志配置 ====================

class LoggingConfig(BaseModel):
    """日志配置模型"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(
        default="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        description="日志格式"
    )
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S", description="日期时间格式")
    file_path: Optional[str] = Field(default=None, description="日志文件路径")
    max_bytes: int = Field(default=10 * 1024 * 1024, description="单个日志文件最大字节数 (10MB)")
    backup_count: int = Field(default=5, description="保留的旧日志文件数量")
    encoding: str = Field(default="utf-8", description="日志文件编码")
    json_format: bool = Field(default=False, description="是否使用结构化 JSON 格式")

    model_config = ConfigDict(extra="ignore")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {levels}")
        return v.upper()

# ==================== 适配器配置基类 ====================

class BaseAdapterConfig(BaseModel):
    """适配器配置基类"""
    adapter_type: str = Field(..., description="适配器类型")
    config: Dict[str, Any] = Field(default_factory=dict, description="适配器特定配置")

    model_config = ConfigDict(extra="ignore")


# ==================== 模块特定配置 ====================

class ASRModuleConfig(BaseModel):
    """ASR 模块通用配置"""
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="采样率")
    channels: int = Field(default=1, ge=1, le=2, description="声道数")

    model_config = ConfigDict(extra="allow")


class TTSModuleConfig(BaseModel):
    """TTS 模块通用配置"""
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="采样率")
    voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="语音名称")
    rate: str = Field(default="+0%", description="语速")
    volume: str = Field(default="+0%", description="音量")

    model_config = ConfigDict(extra="allow")


class VADModuleConfig(BaseModel):
    """VAD 模块通用配置"""
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="语音检测阈值")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="采样率")

    model_config = ConfigDict(extra="allow")


class LLMModuleConfig(BaseModel):
    """LLM 模块通用配置"""
    model_name: str = Field(default="gpt-3.5-turbo", description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=2000, ge=1, description="最大生成令牌数")
    system_prompt: str = Field(default="You are a helpful AI assistant.", description="系统提示词")

    model_config = ConfigDict(extra="allow")


class ProtocolModuleConfig(BaseModel):
    """协议模块通用配置"""
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8765, ge=1, le=65535, description="监听端口")

    model_config = ConfigDict(extra="allow")


# ==================== 适配器配置包装 ====================

class ASRConfig(BaseAdapterConfig):
    """语音识别 (ASR) 配置"""
    pass


class TTSConfig(BaseAdapterConfig):
    """语音合成 (TTS) 配置"""
    pass


class VADConfig(BaseAdapterConfig):
    """语音活动检测 (VAD) 配置"""
    pass


class LLMConfig(BaseAdapterConfig):
    """大语言模型 (LLM) 配置"""
    pass


class ProtocolConfig(BaseAdapterConfig):
    """协议配置"""
    pass

class ModulesConfig(BaseModel):
    """模块集合配置"""
    asr: Optional[ASRConfig] = None
    tts: Optional[TTSConfig] = None
    vad: Optional[VADConfig] = None
    llm: Optional[LLMConfig] = None
    protocols: Optional[ProtocolConfig] = None

class AppConfig(BaseModel):
    """应用全局配置"""
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    modules: ModulesConfig = Field(default_factory=ModulesConfig)
