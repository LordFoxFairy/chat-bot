# app.py (Refactored main application entry point)
import asyncio
import logging
from app_ws_server import WebSocketServer  # 引入重构后的WebSocketServer
from services.config_loader import ConfigLoader  # 用于获取日志级别等全局配置

# 配置全局日志记录器
# 注意：如果app_ws_server.py中的main_async或WebSocketServer.start已经配置了basicConfig，
# 这里的配置可能会被覆盖或冲突，取决于调用顺序。
# 推荐在一个统一的地方进行日志配置，例如这里。
config_temp = ConfigLoader("config.yaml").get_config()
log_level_str = config_temp.get("logging_level", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)  # 获取日志级别

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # 获取根记录器或特定记录器


def main():
    """
    主应用程序入口函数。
    初始化并启动WebSocket服务器。
    """
    logger.info("应用程序启动中...")

    # 从配置文件加载配置，WebSocketServer内部也会加载，但这里可以用于其他全局设置
    config_path = "configs/config.yaml"  # 或者从环境变量、命令行参数获取

    # 创建WebSocketServer实例
    # WebSocketServer的__init__方法会处理其自身的配置加载和模块初始化
    try:
        ws_server = WebSocketServer(config_path=config_path)
    except SystemExit as e:
        logger.fatal(f"无法初始化WebSocket服务器，系统将退出: {e}")
        return  # 退出主函数
    except Exception as e:
        logger.fatal(f"初始化WebSocket服务器时发生未预料的错误: {e}", exc_info=True)
        return

    logger.info("准备启动WebSocket服务器...")
    try:
        # 运行WebSocket服务器的异步start方法
        # WebSocketServer.start() 内部包含 asyncio.Future() 来保持运行
        asyncio.run(ws_server.start())
    except KeyboardInterrupt:
        logger.info("应用程序被用户中断 (Ctrl+C)，正在关闭...")
    except Exception as e:
        logger.fatal(f"应用程序运行时发生致命错误: {e}", exc_info=True)
    finally:
        logger.info("应用程序已关闭。")


if __name__ == "__main__":
    main()
