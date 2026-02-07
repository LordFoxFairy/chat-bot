"""
Chat Bot CLI entry point.
Provides command-line interface for starting the chat bot server and desktop app.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_server() -> None:
    """Start the chat bot WebSocket server."""
    # Add project root to path
    project_root = get_project_root()
    sys.path.insert(0, str(project_root))

    from src.utils.config_loader import ConfigLoader
    from src.utils.logging_setup import logger
    from src.core.engine.chat_engine import ChatEngine
    from src.core.session.session_manager import SessionManager, InMemoryStorage

    async def start_server() -> None:
        logger.info("--- Chat Bot Server Starting ---")

        config_path = project_root / "configs" / "config.yaml"

        try:
            # Load configuration
            config = await ConfigLoader.load_config(str(config_path))
            if not config:
                logger.critical(f"Failed to load config from '{config_path}'")
                return

            # Create SessionManager
            storage_backend = InMemoryStorage(maxsize=10000)
            session_manager = SessionManager(storage_backend=storage_backend)

            # Create ChatEngine
            chat_engine = ChatEngine(config=config, session_manager=session_manager)

            # Initialize all modules
            await chat_engine.initialize()

            logger.info("[Server] Ready and waiting for client connections...")

        except Exception as e:
            logger.error(f"Server startup error: {e}", exc_info=True)
            sys.exit(1)

    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)


def run_desktop() -> None:
    """Start the Tauri desktop application."""
    project_root = get_project_root()
    desktop_dir = project_root / "desktop"

    if not desktop_dir.exists():
        print(f"Error: Desktop directory not found at {desktop_dir}")
        sys.exit(1)

    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        # Run npm dev in desktop directory
        subprocess.run(
            ["npm", "run", "dev"],
            cwd=str(desktop_dir),
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
    """Run development mode with both server and desktop."""
    project_root = get_project_root()
    scripts_dir = project_root / "scripts"
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
    """Main CLI entry point."""
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

    # Store config overrides in environment for use by server
    if args.config:
        os.environ["CHATBOT_CONFIG"] = args.config
    if args.host:
        os.environ["CHATBOT_HOST"] = args.host
    if args.port:
        os.environ["CHATBOT_PORT"] = str(args.port)

    # Execute command
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
