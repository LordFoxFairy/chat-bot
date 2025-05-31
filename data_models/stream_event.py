from enum import Enum
from typing import Union, Optional, Any, Dict
from pydantic import BaseModel, Field
from .audio_data import AudioData # 假設 AudioData 在同級目錄
from .text_data import TextData   # 假設 TextData 在同級目錄
import time

class EventType(Enum):
    """
    事件類型枚舉。
    定義了系統中可能發生的各種事件類型。
    """
    # --- 客戶端 (Client) -> 服務端 (Server) ---
    MODULE_ERROR = None
    USER_INPUT_RECEIVED = None
    CLIENT_AUDIO_CHUNK = "client_audio_chunk"     # 客戶端發送音頻塊 (流式)
    CLIENT_AUDIO_BLOCK = "client_audio_block"     # 客戶端發送完整音頻塊 (非流式)
    CLIENT_TEXT_INPUT = "client_text_input"       # 客戶端發送文本輸入
    CLIENT_CONTROL_COMMAND = "client_control_command" # 客戶端發送控制命令 (例如：開始、停止、重置、更改配置)
    CLIENT_HEARTBEAT = "client_heartbeat"         # 客戶端心跳

    # --- 服務端 (Server) -> 客戶端 (Client) ---
    SERVER_TEXT_RESPONSE = "server_text_response"   # 服務端發送最終文本響應 (例如 LLM 的完整回復)
    SERVER_AUDIO_RESPONSE = "server_audio_response"  # 服務端發送最終音頻響應 (例如 TTS 的完整音頻)
    SERVER_TEXT_CHUNK = "server_text_chunk"       # 服務端流式傳輸文本塊 (例如 LLM 的中間結果)
    SERVER_AUDIO_CHUNK = "server_audio_chunk"       # 服務端流式傳輸音頻塊 (例如 TTS 的中間結果)
    SERVER_SPEECH_STATE = "server_speech_state"     # 服務端指示 VAD 語音活動狀態 (例如：speaking, not_speaking, maybe_speaking)
    SERVER_SYSTEM_MESSAGE = "server_system_message" # 服務端發送系統級消息 (例如：提示、警告、信息)
    SERVER_ERROR_MESSAGE = "server_error_message"   # 服務端發送錯誤消息
    SERVER_SESSION_UPDATE = "server_session_update" # 服務端發送會話狀態更新 (例如：用戶ID、會話配置)
    SERVER_MODULE_STATE = "server_module_state"     # 服務端發送特定模塊的狀態 (例如：ASR準備就緒、TTS正在合成)
    SERVER_HEARTBEAT_ACK = "server_heartbeat_ack"   # 服務端心跳確認

    # --- 系統內部事件 (Internal System Events) ---
    # 這些事件通常在服務器內部模塊之間傳遞，不一定直接發送給客戶端
    SYSTEM_VAD_RESULT = "system_vad_result"             # VAD 模塊產生檢測結果 (語音開始/結束/片段)
    SYSTEM_ASR_RESULT_INTERMEDIATE = "system_asr_result_intermediate" # ASR 模塊產生中間識別結果
    SYSTEM_ASR_RESULT_FINAL = "system_asr_result_final"         # ASR 模塊產生最終識別結果
    SYSTEM_LLM_OUTPUT_CHUNK = "system_llm_output_chunk"       # LLM 模塊產生流式輸出塊
    SYSTEM_LLM_OUTPUT_FINAL = "system_llm_output_final"       # LLM 模塊產生最終輸出
    SYSTEM_TTS_AUDIO_CHUNK = "system_tts_audio_chunk"        # TTS 模塊產生流式音頻塊
    SYSTEM_TTS_AUDIO_FINAL = "system_tts_audio_final"        # TTS 模塊產生最終完整音頻
    SYSTEM_PIPELINE_START = "system_pipeline_start"       # 流水線開始處理
    SYSTEM_PIPELINE_END = "system_pipeline_end"           # 流水線處理結束
    SYSTEM_MODULE_ERROR = "system_module_error"           # 某個模塊發生錯誤
    SYSTEM_SESSION_START = "system_session_start"         # 會話開始
    SYSTEM_SESSION_END = "system_session_end"             # 會話結束
    SYSTEM_USER_INTENT = "system_user_intent"             # (可選) NLU/DM 模塊識別出的用戶意圖
    SYSTEM_TOOL_CALL_REQUEST = "system_tool_call_request"   # (可選) LLM 請求調用工具
    SYSTEM_TOOL_CALL_RESPONSE = "system_tool_call_response" # (可選) 工具執行後的響應

    # --- 更通用的控制事件類型 (如果不再使用 ControlSignal 類) ---
    # 可以用這些來替代原來的 SignalType，並將具體信號類型放在 StreamEvent.data 中
    CONTROL_EVENT_SPEECH_START = "control_event_speech_start" # 替代 SignalType.SPEECH_START
    CONTROL_EVENT_SPEECH_END = "control_event_speech_end"     # 替代 SignalType.SPEECH_END
    CONTROL_EVENT_STREAM_START = "control_event_stream_start" # 替代 SignalType.STREAM_START
    CONTROL_EVENT_STREAM_END = "control_event_stream_end"       # 替代 SignalType.STREAM_END

class StreamState(Enum):
    """
    流狀態枚舉。
    """
    LISTENING = "listening"     # 正在聆聽
    PROCESSING = "processing"   # 正在處理
    SPEAKING = "speaking"       # LLM 正在生成回复或 TTS 正在播放
    IDLE = "idle"               # 空闲状态

class StreamEvent(BaseModel):
    """
    流事件模型。
    """
    event_type: EventType = Field(..., description="事件的類型。")
    data: Optional[Union[AudioData, TextData, Dict[str, Any], str]] = Field(None, description="事件的有效載荷。")
    session_id: Optional[str] = Field(None, description="用於跟踪上下文的會話 ID。")
    timestamp: float = Field(default_factory=time.time, description="事件創建的時間戳 (紀元秒)。")
    error_message: Optional[str] = Field(None, description="錯誤消息，如果 event_type 是 SERVER_ERROR_MESSAGE 或 SYSTEM_MODULE_ERROR。")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="事件的附加元數據。")

    def __str__(self):
        """
        返回 StreamEvent 對象的字符串表示形式。
        """
        event_type_value = self.event_type.value if self.event_type else "未知事件類型"
        data_type_name = type(self.data).__name__ if self.data is not None else "None"

        return (f"StreamEvent(類型='{event_type_value}', 會話ID='{self.session_id}', "
                f"時間戳={self.timestamp}, 數據類型='{data_type_name}', "
                f"錯誤='{self.error_message}')")