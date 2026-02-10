"""
Chat Bot 统一入口
整合 CLI 和直接启动功能，提供命令行接口
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def run_server() -> None:
    """启动 WebSocket 服务器"""
    project_root = get_project_root()
    sys.path.insert(0, str(project_root))

    from backend.utils.config_loader import ConfigLoader
    from backend.utils.logging_setup import logger
    from backend.core.engine.chat_engine import ChatEngine
    from backend.core.session.session_manager import SessionManager, InMemoryStorage

    async def start_server() -> None:
        logger.info("--- Chat Bot Server Starting ---")

        # 支持环境变量覆盖配置路径
        config_path = os.environ.get(
            "CHATBOT_CONFIG",
            str(project_root / "backend" / "configs" / "config.yaml")
        )

        chat_engine = None
        try:
            # 加载配置
            config = await ConfigLoader.load_config(config_path)
            if not config:
                logger.critical(f"Failed to load config from '{config_path}'")
                return

            # 创建 SessionManager
            storage_backend = InMemoryStorage(maxsize=10000)
            session_manager = SessionManager(storage_backend=storage_backend)

            # 创建 ChatEngine
            chat_engine = ChatEngine(config=config, session_manager=session_manager)

            # 初始化所有模块
            await chat_engine.initialize()

            logger.info("[Server] Ready and waiting for client connections...")

        except Exception as e:
            logger.error(f"Server startup error: {e}", exc_info=True)
            if chat_engine:
                await chat_engine.shutdown()
            sys.exit(1)

    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)


def run_desktop() -> None:
    """启动 Tauri 桌面应用"""
    project_root = get_project_root()
    frontend_dir = project_root / "frontend"

    if not frontend_dir.exists():
        print(f"Error: Frontend directory not found at {frontend_dir}")
        sys.exit(1)

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(
            ["npm", "run", "dev"],
            cwd=str(frontend_dir),
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Desktop app failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: npm not found. Please install Node.js.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDesktop app stopped.")


def run_dev() -> None:
    """开发模式：同时启动服务器和桌面应用"""
    project_root = get_project_root()
    scripts_dir = project_root / "backend" / "scripts"
    run_script = scripts_dir / "run_desktop.sh"

    if not run_script.exists():
        print(f"Error: Development script not found at {run_script}")
        sys.exit(1)

    try:
        subprocess.run(["bash", str(run_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Development mode failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDevelopment mode stopped.")


def main() -> None:
    """主入口函数"""
    parser = argparse.ArgumentParser(
        prog="chatbot",
        description="Chat Bot - AI Voice Assistant with ASR, LLM, and TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chatbot                 Start the WebSocket server (default)
  chatbot server          Start the WebSocket server
  chatbot desktop         Start the Tauri desktop app
  chatbot dev             Start development mode (server + desktop)

For more information, visit: https://github.com/thefoxfairy/chat-bot
        """,
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="server",
        choices=["server", "desktop", "dev"],
        help="Command to run (default: server)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--host",
        type=str,
        help="Server host (overrides config)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Server port (overrides config)",
    )

    args = parser.parse_args()

    # 环境变量覆盖
    if args.config:
        os.environ["CHATBOT_CONFIG"] = args.config
    if args.host:
        os.environ["CHATBOT_HOST"] = args.host
    if args.port:
        os.environ["CHATBOT_PORT"] = str(args.port)

    # 执行命令
    if args.command == "server":
        run_server()
    elif args.command == "desktop":
        run_desktop()
    elif args.command == "dev":
        run_dev()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
