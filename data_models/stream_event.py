import base64
import json
import time
from enum import Enum
from typing import Union, Optional, Any, Dict

from pydantic import BaseModel, Field, model_validator

from data_models import AudioData, TextData


class EventType(str, Enum):
    """
    定义了所有在系统中流转的事件类型。
    """
    # 客户端发起的事件
    CLIENT_TEXT_INPUT = "CLIENT_TEXT_INPUT"
    SYSTEM_CLIENT_SESSION_START = "SYSTEM_CLIENT_SESSION_START"
    CLIENT_SPEECH_END = "CLIENT_SPEECH_END"  # 由前端VAD检测到语音结束时发送
    STREAM_END = "STREAM_END"  # 由前端在停止录音或断开连接时发送

    # 服务器发起的事件
    SERVER_TEXT_RESPONSE = "SERVER_TEXT_RESPONSE"
    SERVER_AUDIO_RESPONSE = "SERVER_AUDIO_RESPONSE"
    SERVER_SYSTEM_MESSAGE = "SERVER_SYSTEM_MESSAGE"
    SYSTEM_SERVER_SESSION_START = "SYSTEM_SERVER_SESSION_START"

    # 新增：用于实时显示ASR（语音识别）结果的事件
    ASR_UPDATE = "ASR_UPDATE"

    # 内部事件（可选，但有助于区分）
    ASR_RESULT = "asr_result"  # AudioConsumer 内部处理完成的事件
    LLM_START = "llm_start"  # 通知客户端LLM开始思考
    LLM_RESPONSE = "llm_response"  # LLM的流式文本响应


class StreamState(str, Enum):
    """
    流状态枚举。
    """
    LISTENING = "聆听中"
    PROCESSING = "处理中"
    SPEAKING = "说话中"
    IDLE = "空闲中"


class StreamEvent(BaseModel):
    """
    一個完整的、支持輸入和輸出時使用外部辨識符的事件模型。
    """
    event_type: EventType
    event_data: Union[TextData, AudioData, Any] = Field(default=None)
    tag_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    timestamp: float = Field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="其他元数据")


    @model_validator(mode='before')
    @classmethod
    def dispatch_event_data_parsing(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """根据 event_type 解析 event_data 为对应的模型"""
        event_type = data.get('event_type')
        event_data_payload = data.get('event_data')

        if not event_type or not isinstance(event_data_payload, dict):
            return data

        if event_type == EventType.CLIENT_TEXT_INPUT:
            data['event_data'] = TextData.model_validate(event_data_payload)

        return data

    def to_json(self) -> str:
        """序列化为 JSON 字符串

        如果 event_data 是 AudioData，会将 bytes 转为 Base64 编码
        """
        event_dict = self.model_dump()

        # 如果是音频数据，需要 Base64 编码
        if isinstance(self.event_data, AudioData) and self.event_data.data:
            event_dict['event_data']['data'] = base64.b64encode(
                self.event_data.data
            ).decode('utf-8')

        return json.dumps(event_dict)
