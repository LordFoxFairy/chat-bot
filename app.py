# chat-bot/app.py
import asyncio
import os
import sys

from src.utils.config_loader import ConfigLoader
from src.utils.logging_setup import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.engine.chat_engine import ChatEngine
from src.core.session.session_manager import SessionManager, InMemoryStorage


async def main():
    """
    主函数，用于初始化和启动聊天机器人应用。
    """
    logger.info("--- 应用程序启动中 ---")

    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')

    try:
        # 1. 加载配置
        config = await ConfigLoader.load_config(config_path)
        if not config:
            logger.critical(f"错误: 从 '{config_path}' 加载配置失败。")
            return

        # 2. 创建 SessionManager
        storage_backend = InMemoryStorage(maxsize=10000)
        session_manager = SessionManager(storage_backend=storage_backend)

        # 3. 创建 ChatEngine（总负责人，内部创建 ConversationManager）
        chat_engine = ChatEngine(config=config, session_manager=session_manager)

        # 4. 初始化所有模块
        await chat_engine.initialize()

        logger.info("[服务器] 主异步循环将运行，等待客户端连接和消息...")

    except Exception as e:
        logger.error(f"错误: 应用程序启动或运行期间发生错误: {e}", exc_info=True)
        sys.exit(1)

    logger.info("--- 应用程序已停止 ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n应用程序已通过 Ctrl+C 停止。")
    except Exception as e:
        logger.info(f"应用程序在顶层发生未捕获的错误: {e}")
