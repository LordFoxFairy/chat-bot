"""Chat-Bot 应用入口"""

import asyncio
import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.engine.chat_engine import ChatEngine
from src.core.session.session_manager import InMemoryStorage, SessionManager
from src.utils.config_loader import ConfigLoader
from src.utils.logging_setup import logger


async def main() -> None:
    """主函数，初始化和启动聊天机器人应用"""
    logger.info("--- 应用程序启动中 ---")

    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    chat_engine = None

    try:
        # 1. 加载配置
        config = await ConfigLoader.load_config(config_path)
        if not config:
            logger.critical(f"从 '{config_path}' 加载配置失败")
            return

        # 2. 创建 SessionManager
        storage_backend = InMemoryStorage(maxsize=10000)
        session_manager = SessionManager(storage_backend=storage_backend)

        # 3. 创建 ChatEngine
        chat_engine = ChatEngine(config=config, session_manager=session_manager)

        # 4. 初始化所有模块
        await chat_engine.initialize()

        logger.info("[服务器] 等待客户端连接...")

    except Exception as e:
        logger.error(f"应用程序启动失败: {e}", exc_info=True)
        if chat_engine:
            await chat_engine.shutdown()
        sys.exit(1)

    logger.info("--- 应用程序已停止 ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n应用程序已通过 Ctrl+C 停止。")
    except Exception as e:
        logger.info(f"应用程序在顶层发生未捕获的错误: {e}")
