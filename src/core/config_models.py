from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import logging

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

class BaseAdapterConfig(BaseModel):
    """适配器配置基类"""
    adapter_type: str = Field(..., description="适配器类型")
    config: Dict[str, Any] = Field(default_factory=dict, description="适配器特定配置")

    model_config = ConfigDict(extra="ignore")

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
