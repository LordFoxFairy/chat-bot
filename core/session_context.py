import time
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class SessionContext(BaseModel):
    """
    優化後的會話上下文模型。

    用於存儲和傳遞與單個用戶會話相關的狀態、配置和依賴模塊。
    """
    tag_id: Optional[str] = Field(default=None, description="用戶唯一標識。")
    session_id: Optional[str] = Field(default=None, description="用戶會話ID，通常在會話開始時生成。")
    dialogues: Optional[List] = Field(default=None, description="歷史對話")

    audio: Optional[List[bytes]] = Field(default=[], description="音頻數據")
    last_audio_activity_time: Optional[float] = Field(default=None, description="最近一次音频活动时间")

    config: Dict[str, Any] = Field(default_factory=dict,
                                   description="配置信息，例如 API keys 或功能開關。"
                                               "使用 default_factory=dict 避免所有實例共享同一個字典。")

    global_module_manager: Optional[Any] = Field(default=None, description="全局模塊管理器，在整個應用生命週期內共享。")

    module_manager: Optional[Any] = Field(default=None, description="當前會話特定的模塊管理器。")

    # # 消费者相关的配置，作为 SessionContext 的字段
    # consumer_process_interval_seconds: float = Field(0.5, description="消费者检查新数据的频率（秒）。")
    # consumer_audio_segment_threshold_seconds: float = Field(1.5, description="累积音频达到多少秒才开始判断静音。")
    # consumer_silence_timeout_seconds: float = Field(2.0, description="VAD连续静音多长时间后认为用户停止说话。")
    #
    # # 消费者实例本身
    # consumer: Optional[SimpleAudioConsumer] = Field(None, exclude=True)

    class Config:
        # 允許欄位是自訂類物件
        arbitrary_types_allowed = True

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.consumer = SimpleAudioConsumer(
    #         user_context=self,  # 消费者现在可以直接访问这个 SessionContext 实例
    #         process_interval_seconds=self.consumer_process_interval_seconds,
    #         audio_segment_threshold_seconds=self.consumer_audio_segment_threshold_seconds,
    #         silence_timeout_seconds=self.consumer_silence_timeout_seconds
    #     )
    #     self.consumer.start()  # 自动启动消费者任务

    def record_speech_activity(self):
        self._last_speech_activity_time = time.time()

    def get_last_speech_activity_time(self) -> Optional[float]:
        return self._last_speech_activity_time

    def clear_audio_buffer(self):
        self.audio.clear()
        print(f"  [Context {self.session_id}] 音频缓冲区已清空。")

    # def stop_consumer(self):
    #     """停止会话关联的消费者任务。"""
    #     if self.consumer:
    #         self.consumer.stop()
    #         print(f"[SessionContext {self.session_id}] 已停止其消费者任务。")


SessionContext.model_rebuild()
