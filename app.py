# app_ws_server.py
import asyncio
import logging
import os
from core.chat_engine import ChatEngine

# 基本日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app_ws_server_main")

# WebSocket 服务器配置 (可以从 ChatEngine 的全局配置中读取，或在此处硬编码/从环境变量读取)
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765
CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "configs", # 假设配置文件在与此脚本同级的 configs 目录下
    "config.yaml"
)


async def main():
    """
    应用程序主入口点。
    调用 ChatEngine.create_and_run 来启动整个服务。
    """
    logger.info("应用程序启动...")
    logger.info(f"将使用配置文件: {CONFIG_FILE_PATH}")
    logger.info(f"WebSocket 服务器将在 ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT} 上运行")

    await ChatEngine.create_and_run(
        config_file_path=CONFIG_FILE_PATH,
        host=WEBSOCKET_HOST,
        port=WEBSOCKET_PORT
    )
    logger.info("应用程序已结束。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("应用程序通过 KeyboardInterrupt 强制退出。")
    except Exception as e:
        logger.critical(f"应用程序顶层发生未处理的错误: {e}", exc_info=True)
    finally:
        logger.info("应用程序主程序关闭。")