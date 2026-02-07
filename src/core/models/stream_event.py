"""流事件模型

定义系统中所有事件类型和流状态。
"""

import base64
import json
import time
from enum import Enum
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator

from src.core.models.audio_data import AudioData
from src.core.models.text_data import TextData


class EventType(str, Enum):
    """定义系统中流转的事件类型"""

    # 客户端发起的事件
    CLIENT_TEXT_INPUT = "CLIENT_TEXT_INPUT"
    SYSTEM_CLIENT_SESSION_START = "SYSTEM_CLIENT_SESSION_START"
    CLIENT_SPEECH_END = "CLIENT_SPEECH_END"
    STREAM_END = "STREAM_END"

    # 服务器发起的事件
    SERVER_TEXT_RESPONSE = "SERVER_TEXT_RESPONSE"
    SERVER_AUDIO_RESPONSE = "SERVER_AUDIO_RESPONSE"
    SERVER_SYSTEM_MESSAGE = "SERVER_SYSTEM_MESSAGE"
    SYSTEM_SERVER_SESSION_START = "SYSTEM_SERVER_SESSION_START"

    # ASR 事件
    ASR_UPDATE = "ASR_UPDATE"
    ASR_RESULT = "asr_result"

    # LLM 事件
    LLM_START = "llm_start"
    LLM_RESPONSE = "llm_response"

    # 错误事件
    ERROR = "error"


class StreamState(str, Enum):
    """流状态枚举"""

    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    IDLE = "idle"


# 事件类型到数据类型的映射
EVENT_DATA_TYPE_MAP: Dict[EventType, Type[BaseModel]] = {
    EventType.CLIENT_TEXT_INPUT: TextData,
    EventType.SERVER_TEXT_RESPONSE: TextData,
    EventType.ASR_UPDATE: TextData,
    EventType.ASR_RESULT: TextData,
    EventType.LLM_RESPONSE: TextData,
    EventType.SERVER_AUDIO_RESPONSE: AudioData,
}


class StreamEvent(BaseModel):
    """流事件模型

    统一的事件封装，支持多种数据类型。
    """

    event_type: EventType
    event_data: Optional[Union[TextData, AudioData]] = Field(default=None)
    tag_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    timestamp: float = Field(default_factory=time.time)
    state: Optional[StreamState] = Field(default=None, description="当前流状态")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")

    model_config = {"use_enum_values": True}

    @model_validator(mode='before')
    @classmethod
    def dispatch_event_data_parsing(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """根据 event_type 自动解析 event_data"""
        if not isinstance(data, dict):
            return data

        event_type_raw = data.get('event_type')
        event_data_payload = data.get('event_data')

        if not event_type_raw or event_data_payload is None:
            return data

        # 转换事件类型
        try:
            event_type = EventType(event_type_raw) if isinstance(event_type_raw, str) else event_type_raw
        except ValueError:
            return data

        # 如果 event_data 已经是模型实例，直接返回
        if isinstance(event_data_payload, (TextData, AudioData)):
            return data

        # 根据事件类型解析数据
        if isinstance(event_data_payload, dict):
            expected_type = EVENT_DATA_TYPE_MAP.get(event_type)
            if expected_type:
                # 处理 AudioData 的 Base64 解码
                if expected_type == AudioData and 'data' in event_data_payload:
                    raw_data = event_data_payload.get('data')
                    if isinstance(raw_data, str):
                        try:
                            event_data_payload['data'] = base64.b64decode(raw_data)
                        except Exception:
                            pass  # 保持原样

                data['event_data'] = expected_type.model_validate(event_data_payload)

        return data

    def to_json(self) -> str:
        """序列化为 JSON 字符串

        AudioData.data 会被 Base64 编码
        """
        event_dict = self.model_dump()

        # 音频数据 Base64 编码
        if isinstance(self.event_data, AudioData) and self.event_data.data:
            event_dict['event_data']['data'] = base64.b64encode(
                self.event_data.data
            ).decode('utf-8')

        return json.dumps(event_dict, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "StreamEvent":
        """从 JSON 字符串反序列化

        Args:
            json_str: JSON 格式的事件数据

        Returns:
            StreamEvent 实例
        """
        data = json.loads(json_str)
        return cls.model_validate(data)

    @classmethod
    def create_text_event(
        cls,
        event_type: EventType,
        text: str,
        session_id: Optional[str] = None,
        is_final: bool = True,
        **kwargs
    ) -> "StreamEvent":
        """创建文本事件的便捷方法"""
        return cls(
            event_type=event_type,
            event_data=TextData(text=text, is_final=is_final),
            session_id=session_id,
            **kwargs
        )

    @classmethod
    def create_audio_event(
        cls,
        data: bytes,
        session_id: Optional[str] = None,
        is_final: bool = False,
        **kwargs
    ) -> "StreamEvent":
        """创建音频事件的便捷方法"""
        from src.core.models.audio_data import AudioFormat
        return cls(
            event_type=EventType.SERVER_AUDIO_RESPONSE,
            event_data=AudioData(data=data, format=AudioFormat.PCM, is_final=is_final),
            session_id=session_id,
            **kwargs
        )

    @classmethod
    def create_error_event(
        cls,
        error_message: str,
        session_id: Optional[str] = None,
        error_code: Optional[str] = None,
        **kwargs
    ) -> "StreamEvent":
        """创建错误事件"""
        return cls(
            event_type=EventType.ERROR,
            event_data=TextData(text=error_message, is_final=True),
            session_id=session_id,
            metadata={"error_code": error_code} if error_code else {},
            **kwargs
        )
