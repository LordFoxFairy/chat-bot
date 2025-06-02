# core_framework/session_manager.py
import asyncio
import logging
import time
import uuid
from typing import Dict, Optional, Any, List

# 导入 SessionContext，它存储单个会话的数据
# 假设 SessionContext 在同级目录或可通过Python路径访问
from .session_context import SessionContext

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TIMEOUT_SECONDS_SM = 30 * 60  # 默认会话超时时间：30分钟


class SessionManager:
    """
    会话管理器 (SessionManager)
    辅助全局 ChatEngine 管理 SessionContext 对象的生命周期。
    负责创建、跟踪、检索和清理 SessionContext 实例。
    """

    def __init__(self,
                 global_config: Dict[str, Any],  # 直接接收加载好的全局配置
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化 SessionManager。

        Args:
            global_config (Dict[str, Any]): 已加载的全局应用配置，将作为创建 SessionContext 时的基础配置。
            loop (Optional[asyncio.AbstractEventLoop]): 事件循环。如果为 None，则获取当前运行的循环。
        """
        self.global_config = global_config
        self.loop = loop if loop else asyncio.get_event_loop()

        self.active_sessions: Dict[str, SessionContext] = {}  # 存储活动会话，键为 session_id

        # 从全局配置中获取会话超时设置
        system_settings = self.global_config.get("system_config", {})
        self.session_timeout_seconds = system_settings.get("session_timeout_seconds",
                                                           DEFAULT_SESSION_TIMEOUT_SECONDS_SM)

        self._cleanup_task: Optional[asyncio.Task] = None  # 用于定期清理超时会话的后台任务
        self.is_shutting_down = False  # 标记是否正在关闭，以优雅地停止清理任务

        logger.info(f"SessionManager 初始化完成。会话超时设置为: {self.session_timeout_seconds} 秒。")

    def start_cleanup_task(self):
        """启动会话超时清理任务 (如果尚未运行)。"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = self.loop.create_task(self._periodic_session_cleanup())
            logger.info("SessionManager 的会话超时清理任务已启动。")

    def create_session_context(self,
                               websocket: Any,  # WebSocket 连接对象
                               external_session_id: Optional[str] = None,
                               user_specific_config: Optional[Dict[str, Any]] = None
                               ) -> SessionContext:
        """
        创建并返回一个新的 SessionContext 实例。
        此方法不负责异步初始化 SessionContext 内部可能包含的组件（例如 ChatEngine 的特定会话部分）。
        它主要用于创建 SessionContext 数据容器。

        Args:
            websocket (Any): 与此会话关联的 WebSocket 连接对象。
            external_session_id (Optional[str]): 外部提供的会话ID (例如客户端通过WebSocket传递的UUID)。
            user_specific_config (Optional[Dict[str, Any]]): 此用户的特定配置覆盖。

        Returns:
            SessionContext: 新创建的 SessionContext 对象。

        Raises:
            ValueError: 如果提供的 external_session_id 已被占用。
        """
        session_id_to_use: str
        if external_session_id:
            if external_session_id in self.active_sessions:
                logger.error(f"尝试使用已存在的 external_session_id '{external_session_id}' 创建新会话。")
                raise ValueError(f"会话ID '{external_session_id}' 已被占用。")
            session_id_to_use = external_session_id
        else:
            session_id_to_use = str(uuid.uuid4())

        logger.info(
            f"正在为 WebSocket 连接 {websocket.remote_address if hasattr(websocket, 'remote_address') else ''} 创建新的 SessionContext，ID: '{session_id_to_use}'")

        session_ctx = SessionContext(
            session_id=session_id_to_use,
            websocket=websocket,
            global_config=self.global_config,  # 将全局配置作为基础
            user_specific_config=user_specific_config
        )

        self.active_sessions[session_id_to_use] = session_ctx
        logger.info(f"SessionContext '{session_id_to_use}' 已创建并注册。当前活动会话数: {len(self.active_sessions)}")

        # 确保清理任务正在运行
        self.start_cleanup_task()

        return session_ctx

    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """
        通过会话 ID 检索一个活动的 SessionContext。
        如果找到，会更新其最后活动时间。
        """
        session_ctx = self.active_sessions.get(session_id)
        if session_ctx:
            session_ctx.update_activity_time()  # SessionContext 内部方法更新时间戳
            logger.debug(f"SessionContext '{session_id}' 已被访问，最后活动时间已更新。")
        else:
            logger.warning(f"尝试访问不存在或已过期的 SessionContext ID '{session_id}'。")
        return session_ctx

    async def delete_session_context(self, session_id: str, reason: str = "未知原因"):
        """
        删除一个指定的 SessionContext 并清理其资源 (主要是关闭其关联的 WebSocket)。
        """
        session_ctx = self.active_sessions.pop(session_id, None)
        if session_ctx:
            logger.info(f"正在删除 SessionContext '{session_id}'，原因: {reason}...")
            try:
                await session_ctx.close(reason=reason)  # 调用 SessionContext 的 close 方法
            except Exception as e:
                logger.error(f"关闭 SessionContext '{session_id}' 时发生错误: {e}", exc_info=True)
            logger.info(f"SessionContext '{session_id}' 已成功删除。当前活动会话数: {len(self.active_sessions)}")
        else:
            logger.warning(f"尝试删除不存在的 SessionContext ID '{session_id}'。")

    async def _periodic_session_cleanup(self):
        """
        定期运行的后台任务，用于检查并移除超时的 SessionContext。
        """
        logger.info("SessionManager 的会话超时定期清理任务正在运行...")
        while not self.is_shutting_down:  # 允许通过标志优雅停止
            check_interval = max(10, min(60, self.session_timeout_seconds // 5))
            if not self.active_sessions:
                check_interval = max(60, self.session_timeout_seconds // 2)

            try:
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                logger.info("会话清理任务被取消。")
                break  # 退出循环

            if self.is_shutting_down:
                break  # 再次检查，因为 sleep 可能被中断

            now = time.time()
            expired_session_ids: List[str] = []
            current_sessions_snapshot = list(self.active_sessions.items())

            for sid, session_obj in current_sessions_snapshot:
                if (now - session_obj.last_activity_time) > self.session_timeout_seconds:
                    expired_session_ids.append(sid)

            if expired_session_ids:
                logger.info(f"发现 {len(expired_session_ids)} 个超时会话: {expired_session_ids}。正在清理...")
                for sid_to_delete in expired_session_ids:
                    if sid_to_delete in self.active_sessions:
                        await self.delete_session_context(sid_to_delete, reason="超时")
            else:
                logger.debug(f"定期清理：未发现超时会话。当前活动会话数: {len(self.active_sessions)}")
        logger.info("SessionManager 的会话超时清理任务已停止。")

    async def shutdown(self):
        """
        关闭 SessionManager。
        这包括停止后台清理任务并关闭所有仍然活动的会话。
        """
        logger.info("SessionManager 正在关闭，开始清理所有资源...")
        self.is_shutting_down = True  # 设置标志以停止清理任务的循环

        if self._cleanup_task and not self._cleanup_task.done():
            logger.info("正在取消会话超时定期清理任务...")
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.info("会话超时定期清理任务已成功取消。")
            except Exception as e:
                logger.error(f"等待会话清理任务结束时发生错误: {e}", exc_info=True)
            self._cleanup_task = None

        active_session_ids_to_close = list(self.active_sessions.keys())
        if active_session_ids_to_close:
            logger.info(f"正在关闭 {len(active_session_ids_to_close)} 个剩余的活动会话: {active_session_ids_to_close}")
            closing_tasks = [
                self.delete_session_context(session_id, reason="SessionManager 关闭")
                for session_id in active_session_ids_to_close
            ]
            if closing_tasks:
                results = await asyncio.gather(*closing_tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"关闭会话 {active_session_ids_to_close[i]} 时发生错误: {result}")
        else:
            logger.info("没有活动的会话需要关闭。")

        self.active_sessions.clear()
        logger.info("SessionManager 关闭完成。所有活动会话已清理。")
