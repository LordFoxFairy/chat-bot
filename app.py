# chat-bot/app.py
import asyncio
import os
import sys

from utils.config_loader import ConfigLoader
from utils.logging_setup import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.chat_engine import ChatEngine


async def main():
    """
    主函数，用于初始化和启动聊天机器人应用。
    """
    logger.info("--- 应用程序启动中 ---")

    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'config.yaml')

    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)
        if not config:
            logger.critical(f"错误: 从 '{config_path}' 加载配置失败。")
            return
        await ChatEngine(config).initialize()
        logger.info("[服务器] 主异步循环将运行，等待客户端连接和消息...")

    except Exception as e:
        logger.info(f"错误: 应用程序启动或运行期间发生错误: {e}")
        sys.exit(1)

    logger.info("--- 应用程序已停止 ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n应用程序已通过 Ctrl+C 停止。")
    except Exception as e:
        logger.info(f"应用程序在顶层发生未捕获的错误: {e}")
