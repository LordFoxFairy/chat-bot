"""
多轮对话上下文保持测试

测试目的:
- WebSocket 协议正常工作
- LLM 对话上下文保持
- 多轮对话流程

测试步骤:
1. 启动完整的 ChatEngine（包含所有模块）
2. 通过 WebSocket 连接
3. 第一轮对话: 发送"我叫小明"，等待回复
4. 第二轮对话: 发送"我刚才说我叫什么？"，验证 LLM 记住了上下文
5. 验证第二轮回复中包含"小明"
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Optional

import websockets
from dotenv import load_dotenv

# 添加项目根目录到 pythonpath
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.engine.chat_engine import ChatEngine
from backend.core.models import StreamEvent, EventType, TextData
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.utils.config_loader import ConfigLoader
from backend.utils.logging_setup import setup_logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ContextMemoryTest")

# 全局变量
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 28765  # 使用不同端口避免冲突
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}"


class ChatEngineTestServer:
    """测试用 ChatEngine 服务器"""

    def __init__(self):
        self.chat_engine: Optional[ChatEngine] = None
        self.server_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动 ChatEngine 服务器"""
        logger.info("正在启动测试服务器...")

        # 1. 加载配置
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "backend", "configs", "config.yaml")
        config = await ConfigLoader.load_config(config_path)

        # 修改配置以适应测试
        # 删除 TTS, VAD, ASR 配置以简化测试 (只测试 LLM 上下文)
        if "modules" in config:
            # 直接删除不需要的模块配置（因为 ChatEngine 没有检查 enabled 字段）
            config["modules"].pop("tts", None)
            config["modules"].pop("vad", None)
            config["modules"].pop("asr", None)

            # 修改协议端口
            if "protocols" in config["modules"]:
                protocol_config = config["modules"]["protocols"].get("config", {})
                if "websocket" in protocol_config:
                    protocol_config["websocket"]["port"] = SERVER_PORT
                    protocol_config["websocket"]["host"] = SERVER_HOST

        # 2. 初始化组件
        setup_logging({"level": "INFO"})
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)

        self.chat_engine = ChatEngine(config=config, session_manager=session_manager)
        await self.chat_engine.initialize()

        # 3. 启动 WebSocket 服务
        protocol = self.chat_engine.protocol_modules.get("protocols")
        if not protocol:
            raise RuntimeError("WebSocket 协议未初始化")

        # 在后台启动服务器
        self.server_task = asyncio.create_task(protocol.start())

        # 等待服务器启动
        await asyncio.sleep(1)
        logger.info(f"测试服务器已启动: {WS_URL}")

    async def stop(self):
        """停止服务器"""
        logger.info("正在关闭测试服务器...")

        # 先取消 server_task（避免 chat_engine.shutdown() 中的死锁）
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()

        if self.chat_engine:
            try:
                await asyncio.wait_for(self.chat_engine.shutdown(), timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning("ChatEngine shutdown timeout")
            except Exception as e:
                logger.warning(f"ChatEngine shutdown error: {e}")

        # 等待 server_task 完成
        if self.server_task:
            try:
                await asyncio.wait_for(self.server_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        logger.info("测试服务器已关闭")


async def send_and_receive(
    ws,
    session_id: str,
    tag_id: str,
    message: str,
    timeout: float = 30.0
) -> str:
    """发送消息并接收完整回复

    Args:
        ws: WebSocket 连接
        session_id: 会话 ID
        tag_id: 标签 ID
        message: 要发送的消息
        timeout: 超时时间（秒）

    Returns:
        完整的回复文本
    """
    # 发送消息
    input_event = StreamEvent(
        event_type=EventType.CLIENT_TEXT_INPUT,
        session_id=session_id,
        tag_id=tag_id,
        event_data=TextData(text=message, is_final=True)
    )
    await ws.send(input_event.to_json())
    logger.info(f"发送消息: {message}")

    # 接收回复
    response_text = ""
    start_time = asyncio.get_event_loop().time()

    while True:
        # 检查超时
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            logger.warning(f"等待回复超时 ({timeout}s)")
            break

        try:
            response = await asyncio.wait_for(ws.recv(), timeout=5.0)
            event = StreamEvent.from_json(response)

            if event.event_type == EventType.SERVER_TEXT_RESPONSE:
                text_data = event.event_data
                # text_data 可能是 TextData 对象或字典
                if isinstance(text_data, TextData):
                    text = text_data.text
                    is_final = text_data.is_final
                elif isinstance(text_data, dict):
                    text = text_data.get("text", "")
                    is_final = text_data.get("is_final", False)
                else:
                    text = ""
                    is_final = False

                if text:
                    response_text += text
                    print(text, end="", flush=True)

                # 检查是否为最终回复
                if is_final:
                    print()  # 换行
                    break

            elif event.event_type == EventType.ERROR:
                logger.error(f"收到错误: {event.event_data}")
                break

        except asyncio.TimeoutError:
            # 如果已经收到一些内容，认为对话结束
            if response_text:
                print()
                break
            continue

    return response_text


async def run_context_memory_test() -> bool:
    """运行上下文保持测试

    Returns:
        测试是否通过
    """
    server = ChatEngineTestServer()

    try:
        # 启动服务器
        await server.start()

        # 连接 WebSocket
        logger.info(f"正在连接到 {WS_URL}...")
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            logger.info("WebSocket 连接建立")

            # 1. 发送注册事件
            tag_id = f"test_client_{uuid.uuid4().hex[:8]}"
            register_event = StreamEvent(
                event_type=EventType.SYSTEM_CLIENT_SESSION_START,
                tag_id=tag_id
            )
            await ws.send(register_event.to_json())
            logger.info("发送注册请求")

            # 等待会话确认
            session_id = None
            while not session_id:
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                event = StreamEvent.from_json(response)
                if event.event_type == EventType.SYSTEM_SERVER_SESSION_START:
                    session_id = event.session_id
                    logger.info(f"收到会话确认: session_id={session_id}")
                elif event.event_type == EventType.ERROR:
                    logger.error(f"注册失败: {event.event_data}")
                    return False

            # 2. 第一轮对话: 发送"我叫小明"
            logger.info("\n" + "="*50)
            logger.info("[Round 1] 开始第一轮对话")
            logger.info("="*50)
            response1 = await send_and_receive(
                ws, session_id, tag_id,
                "我叫小明",
                timeout=30.0
            )
            logger.info(f"[Round 1] 完整回复: {response1}")

            # 短暂等待
            await asyncio.sleep(0.5)

            # 3. 第二轮对话: 发送"我刚才说我叫什么？"
            logger.info("\n" + "="*50)
            logger.info("[Round 2] 开始第二轮对话")
            logger.info("="*50)
            response2 = await send_and_receive(
                ws, session_id, tag_id,
                "我刚才说我叫什么？",
                timeout=30.0
            )
            logger.info(f"[Round 2] 完整回复: {response2}")

            # 4. 验证结果
            logger.info("\n" + "="*50)
            logger.info("测试结果验证")
            logger.info("="*50)

            if "小明" in response2:
                logger.info("SUCCESS: 上下文保持正常，LLM 记住了名字 '小明'")
                return True
            else:
                logger.error("FAILURE: 上下文丢失，LLM 未能回忆起名字")
                logger.error(f"  期望: 回复中包含 '小明'")
                logger.error(f"  实际: {response2}")
                return False

    except Exception as e:
        logger.error(f"测试过程发生错误: {e}", exc_info=True)
        return False

    finally:
        await server.stop()


async def main():
    """主函数"""
    # 加载环境变量 (需要 API_KEY)
    load_dotenv()

    if not os.getenv("API_KEY"):
        logger.error("未找到 API_KEY 环境变量，无法测试 LLM")
        sys.exit(1)

    logger.info("="*60)
    logger.info("多轮对话上下文保持测试")
    logger.info("="*60)

    success = await run_context_memory_test()

    if success:
        logger.info("\n" + "="*60)
        logger.info("TEST PASSED")
        logger.info("="*60)
        sys.exit(0)
    else:
        logger.info("\n" + "="*60)
        logger.info("TEST FAILED")
        logger.info("="*60)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
