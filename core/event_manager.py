import asyncio
from utils.logging_setup import logger
from typing import Callable, Dict, List, Any, Coroutine, Set, Optional
from data_models.stream_event import StreamEvent, EventType

# 定义监听器回调函数的类型别名，它接收一个 StreamEvent 对象
ListenerCallback = Callable[[StreamEvent], Coroutine[Any, Any, None]]


class EventManager:
    """
    事件管理器 (EventManager)
    负责在应用程序的不同模块之间异步分发和处理事件。
    支持事件队列和直接事件分发两种模式。
    事件统一使用您项目定义的 StreamEvent 对象进行封装。
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化 EventManager。

        Args:
            loop (Optional[asyncio.AbstractEventLoop]): asyncio 事件循环。
                                                        如果为 None，则获取当前正在运行的事件循环。
        """
        # 使用字典来存储事件类型及其对应的监听器集合
        # 键现在是您定义的 EventType 枚举成员
        self.listeners: Dict[EventType, Set[ListenerCallback]] = {}
        self.event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()  # 事件队列，存储 StreamEvent 对象
        self.worker_task: Optional[asyncio.Task] = None  # 事件处理工作任务
        self._loop = loop if loop else asyncio.get_event_loop()
        self._lock = asyncio.Lock()  # 用于保护 listeners 字典的并发访问

        logger.info("EventManager 初始化完成。")

    async def register_listener(self, event_type: EventType, listener: ListenerCallback):
        """
        注册事件监听器。
        一个监听器是一个异步函数，它将接收一个 StreamEvent 对象作为参数。

        Args:
            event_type (EventType): 要监听的事件类型 (使用您定义的 EventType)。
            listener (ListenerCallback): 异步回调函数。
        """
        async with self._lock:  # 获取锁以安全地修改监听器列表
            if event_type not in self.listeners:
                self.listeners[event_type] = set()

            if listener not in self.listeners[event_type]:
                self.listeners[event_type].add(listener)
                # 使用 .value 获取枚举的字符串值用于日志记录
                logger.info(
                    f"监听器 {getattr(listener, '__name__', str(listener))} 已注册到事件类型 '{event_type.value}'")
            else:
                logger.warning(
                    f"监听器 {getattr(listener, '__name__', str(listener))} 已注册过事件类型 '{event_type.value}'")

    async def unregister_listener(self, event_type: EventType, listener: ListenerCallback):
        """
        注销事件监听器。

        Args:
            event_type (EventType): 要注销的事件类型。
            listener (ListenerCallback): 要注销的异步回调函数。
        """
        async with self._lock:  # 获取锁以安全地修改监听器列表
            if event_type in self.listeners and listener in self.listeners[event_type]:
                self.listeners[event_type].remove(listener)
                logger.info(
                    f"监听器 {getattr(listener, '__name__', str(listener))} 已从事件类型 '{event_type.value}' 注销")
                if not self.listeners[event_type]:
                    del self.listeners[event_type]
            else:
                logger.warning(
                    f"监听器 {getattr(listener, '__name__', str(listener))} 未找到对应的事件类型 '{event_type.value}' 或事件类型不存在。")

    async def dispatch_event(self, event: StreamEvent):
        """
        立即异步分发事件给所有注册的监听器。
        此方法不通过内部事件队列，而是直接调用监听器。

        Args:
            event (StreamEvent): 要分发的事件对象 (使用您定义的 StreamEvent 模型)。
        """
        listeners_to_call: List[ListenerCallback] = []
        # 确保 event.event_type 是您定义的 EventType 的实例
        if not isinstance(event.event_type, EventType):
            logger.error(
                f"分发事件时遇到无效的 event_type: {event.event_type} (类型: {type(event.event_type)}). 事件详情: {event}")
            return

        async with self._lock:
            if event.event_type in self.listeners:
                listeners_to_call = list(self.listeners[event.event_type])

        if listeners_to_call:
            # 使用 event.event_type.value 来获取枚举的字符串值进行日志记录
            # StreamEvent 的 __str__ 方法已经提供了良好的日志输出
            logger.debug(f"直接分发事件 {event} 给 {len(listeners_to_call)} 个监听器。")

            tasks = [listener(event) for listener in listeners_to_call]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    listener_name = getattr(listeners_to_call[i], '__name__', str(listeners_to_call[i]))
                    logger.error(f"监听器 {listener_name} 处理事件 {event} 时发生错误: {result}", exc_info=result)
        else:
            logger.debug(f"事件 {event} 没有注册的监听器。")

    async def post_event(self, event: StreamEvent):
        """
        将事件发布到内部队列中，由后台工作任务异步处理和分发。

        Args:
            event (StreamEvent): 要发布的事件对象 (使用您定义的 StreamEvent 模型)。
        """
        if not isinstance(event.event_type, EventType):
            logger.error(
                f"发布事件时遇到无效的 event_type: {event.event_type} (类型: {type(event.event_type)}). 事件详情: {event}")
            return

        await self.event_queue.put(event)
        logger.debug(f"事件 {event} 已发布到队列。")

    async def _event_worker(self):
        """
        后台工作任务，从事件队列中获取事件并分发给相应的监听器。
        此任务会持续运行直到收到关闭信号或被取消。
        """
        logger.info("事件处理工作线程已启动。")
        while True:
            event_retrieved = False  # 标记事件是否已成功从队列中取出
            try:
                event = await self.event_queue.get()
                event_retrieved = True  # 事件已成功取出

                # 检查是否为特殊的关闭信号事件
                if event.event_type == EventType.SERVER_SYSTEM_MESSAGE and event.data == "_shutdown_signal_":
                    logger.info("事件处理工作线程收到关闭信号，正在退出。")
                    self.event_queue.task_done()
                    break

                await self.dispatch_event(event)
                self.event_queue.task_done()

            except asyncio.CancelledError:
                logger.info("事件处理工作线程被取消。")
                # 如果任务被取消时，事件已经从队列中取出但尚未处理完毕，
                # 仍需调用 task_done()，以避免 join() 阻塞。
                if event_retrieved:
                    try:
                        self.event_queue.task_done()
                    except ValueError:  # task_done() 可能已被调用或队列状态异常
                        logger.warning("在事件工作线程的 CancelledError 处理中调用 task_done() 时出现 ValueError。")
                break
            except Exception as e:
                logger.error(f"事件处理工作线程发生错误: {e}", exc_info=True)
                # 如果事件已从队列中取出，但在处理过程中发生错误，
                # 仍需调用 task_done() 以避免队列的 join() 方法永久阻塞。
                if event_retrieved:  # 检查事件是否真的被取出来了
                    try:
                        self.event_queue.task_done()
                    except ValueError:
                        # 这可能发生在 task_done() 被意外多次调用等情况
                        logger.warning("在事件工作线程的异常处理中调用 task_done() 时出现 ValueError。")

    async def start_worker(self):
        """
        启动事件处理工作任务（如果尚未运行）。
        """
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = self._loop.create_task(self._event_worker())
            logger.info("事件管理器的工作任务已创建并启动。")
        else:
            logger.warning("事件管理器的工作任务已在运行中。")

    async def stop_worker(self, timeout: float = 5.0):
        """
        优雅地停止事件处理工作任务。
        会向队列发送一个特殊的关闭信号，并等待工作任务处理完当前事件后退出。

        Args:
            timeout (float): 等待工作任务正常退出的最长时间（秒）。
        """
        if self.worker_task and not self.worker_task.done():
            logger.info("正在尝试停止事件处理工作线程...")

            shutdown_event = StreamEvent(
                event_type=EventType.SERVER_SYSTEM_MESSAGE,
                session_id="system_shutdown"
            )
            await self.post_event(shutdown_event)

            try:
                await asyncio.wait_for(self.worker_task, timeout=timeout)
                logger.info("事件处理工作线程已成功停止。")
            except asyncio.TimeoutError:
                logger.warning(f"事件处理工作线程在 {timeout} 秒内未停止。正在取消任务...")
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    logger.info("事件处理工作线程已被取消。")
            except Exception as e:
                logger.error(f"停止事件处理工作线程时发生错误: {e}", exc_info=True)
            finally:
                self.worker_task = None
        else:
            logger.info("事件处理工作线程未运行或已停止。")
