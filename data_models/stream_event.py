import time
from enum import Enum
from typing import Union, Optional, Any, Dict

from pydantic import BaseModel, Field, model_validator

from data_models import AudioData, TextData


class EventType(str, Enum):
    """
    事件类型枚举。用于标识系统中可能发生的各类事件。
    """

    # 客户端 -> 服务端
    CLIENT_AUDIO_CHUNK = "客户端音频流"
    CLIENT_AUDIO_BLOCK = "客户端完整音频"
    CLIENT_TEXT_INPUT = "CLIENT_TEXT_INPUT"  # 客户端文本输入

    # 服务端 -> 客户端
    SERVER_TEXT_RESPONSE = "SERVER_TEXT_RESPONSE"  # 服务端文本回复
    SERVER_AUDIO_RESPONSE = "SERVER_AUDIO_RESPONSE" # 服务端音频回复
    SERVER_SPEECH_STATE = "SERVER_SPEECH_STATE" # 服务端语音状态
    SERVER_SYSTEM_MESSAGE = "SERVER_SYSTEM_MESSAGE" # 服务端系统消息
    SERVER_ERROR_MESSAGE = "SERVER_ERROR_MESSAGE" # 服务端错误消息
    SERVER_SESSION_UPDATE = "服务端会话更新"
    SERVER_MODULE_STATE = "服务端模块状态"

    # 注冊事件
    SYSTEM_CLIENT_SESSION_START = "SYSTEM_CLIENT_SESSION_START"  # 系统会话申請注冊
    SYSTEM_SERVER_SESSION_START = "SYSTEM_SERVER_SESSION_START"  # 系统会话注冊成功


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

    @model_validator(mode='before')
    @classmethod
    def dispatch_event_data_parsing(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        根據 event_type 的值，手動將 event_data 解析成對應的 Pydantic 模型。
        """
        event_type = data.get('event_type')
        event_data_payload = data.get('event_data')

        if not event_type or not isinstance(event_data_payload, dict):
            # 如果缺少必要信息，則不處理，讓後續的標準驗證去報錯
            return data

        # --- 這就是您的分發邏輯 ---
        if event_type == EventType.CLIENT_TEXT_INPUT:
            # 將原始字典解析為 TextData 物件
            data['event_data'] = TextData.model_validate(event_data_payload)
        else:
            # 如果有未知的 event_type，可以選擇拋出錯誤或保持原樣
            # raise ValueError(f"未知的事件類型: {event_type}")
            pass

        return data
