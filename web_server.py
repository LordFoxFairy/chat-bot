import asyncio
import json
import logging
import os
from typing import Optional

import websockets

# --- 假设这些是您项目的模块 ---
try:
    from services.config_loader import ConfigLoader
    from core_framework.module_manager import ModuleManager
    from core_framework.chat_engine import ChatEngine  # 导入 ChatEngine
    from data_models.text_data import TextData
    from data_models.audio_data import AudioData
    # from modules.base_tts import BaseTTS # ChatEngine 会管理 TTS 模块
except ImportError as e:
    print(f"导入 chat-bot 模块时出错: {e}")
    print("请确保 chat-bot 项目根目录在 PYTHONPATH 中，或调整导入路径。")
    exit(1)
# --- 结束假设的项目模块 ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chat_engine_websocket_server")  # 修改 logger 名称

# 全局变量 (在 main 中初始化)
config_loader_instance: Optional[ConfigLoader] = None  # 修改变量名以示区分
module_manager: Optional[ModuleManager] = None
chat_engine: Optional[ChatEngine] = None  # 添加 ChatEngine 实例

# 服务器配置
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765
CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # 使用 abspath 更可靠
    "configs",
    "config.yaml"
)


async def initialize_chatbot_components():
    """初始化 ConfigLoader, ModuleManager, 和 ChatEngine。"""
    global config_loader_instance, module_manager, chat_engine

    logger.info(f"从以下位置加载配置: {CONFIG_FILE_PATH}")
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.error(f"配置文件未找到: {CONFIG_FILE_PATH}")
        raise FileNotFoundError(f"配置文件未找到: {CONFIG_FILE_PATH}")

    config_loader_instance = ConfigLoader()  # 创建 ConfigLoader 实例
    # ModuleManager 内部会调用 config_loader_instance.load_config(CONFIG_FILE_PATH)

    # 修正 ModuleManager 的实例化方式
    # ModuleManager 的 __init__ 签名是 (self, config_loader: ConfigLoader, config_path: str, ...)
    module_manager = ModuleManager(
        config=config_loader_instance.load_config(CONFIG_FILE_PATH),  # 传递 ConfigLoader 实例
    )

    await module_manager.initialize_modules()

    # 初始化 ChatEngine
    # 假设 ChatEngine 的 __init__ 接收 module_manager
    chat_engine = ChatEngine(
        module_manager=module_manager,
        # 如果 ChatEngine 还需要全局配置字典，可以这样传递：
        # global_config=module_manager.global_config, # ModuleManager 内部加载了 global_config
        # event_loop=asyncio.get_event_loop() # 如果 ChatEngine 需要显式传递 loop
    )
    # 如果 ChatEngine 有异步初始化方法
    if hasattr(chat_engine, 'initialize') and callable(chat_engine.initialize):
        await chat_engine.initialize()

    logger.info("Chatbot 组件 (包括 ChatEngine) 初始化成功。")


async def websocket_connection_handler(websocket: websockets.WebSocketServerProtocol, path: str):
    """处理单个 WebSocket 客户端连接。"""
    client_address = websocket.remote_address
    session_id = f"ws_{websocket.id}"
    logger.info(f"客户端 {client_address} 已连接。会话 ID: {session_id}")

    if not chat_engine:
        logger.error("ChatEngine 未初始化。无法处理连接。")
        await websocket.close(code=1011, reason="服务器内部错误: ChatEngine 未就绪")
        return

    # 假设 ChatEngine 有 on_client_connect 方法
    if hasattr(chat_engine, 'on_client_connect'):
        await chat_engine.on_client_connect(session_id, websocket)

    try:
        async for message in websocket:
            if isinstance(message, str):
                logger.info(f"[{session_id}] 收到文本消息: {message[:100]}")
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    # 确保 ChatEngine 有 process_message 方法来统一处理
                    # 并且该方法能处理不同类型的输入 (文本，音频指示等)
                    # 并通过 websocket 返回数据
                    if msg_type in ["tts_request", "chat_message"]:
                        payload = data.get("text")
                        if payload:
                            # 假设 ChatEngine 的 process_message 或类似方法会处理并直接通过 websocket 发送
                            # 或者返回一个可迭代的音频块流
                            # 为简化，我们假设 ChatEngine 的 process_message 会处理输出
                            # 您需要根据 ChatEngine 的实际接口调整这里的调用
                            # 例如，如果 process_message 返回音频流：
                            # async for audio_chunk in chat_engine.process_message(...):
                            #    await websocket.send(audio_chunk)
                            # await websocket.send("TTS_COMPLETE") # 或由 ChatEngine 发送
                            await chat_engine.process_message(
                                input_payload=payload,
                                session_id=session_id,
                                input_origin_type="text",  # 指示输入源是文本
                            )
                        else:
                            await websocket.send(
                                json.dumps({"type": "error", "message": f"{msg_type} 请求中未提供文本。"}))

                    elif msg_type == "audio_stream_end":
                        logger.info(f"[{session_id}] 客户端发出音频流结束信号。")
                        # 通知 ChatEngine 音频流结束
                        await chat_engine.process_message(
                            input_payload=None,  # 表示流结束
                            session_id=session_id,
                            input_origin_type="audio_end",
                        )
                        # await websocket.send(json.dumps({"type": "info", "message": "音频流结束已确认。"})) # ChatEngine 可能会发送确认
                    else:
                        logger.warning(f"[{session_id}] JSON 中未知的消息类型: {msg_type}")
                        await websocket.send(json.dumps({"type": "error", "message": f"未知的消息类型: {msg_type}"}))

                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] 收到非 JSON 文本消息: {message[:100]}。正在忽略。")
                    await websocket.send(json.dumps({"type": "error", "message": "无效的 JSON 格式。"}))

            elif isinstance(message, bytes):
                logger.debug(f"[{session_id}] 收到二进制音频数据: {len(message)} 字节")
                await chat_engine.process_message(
                    input_payload=message,  # 音频字节
                    session_id=session_id,
                    input_origin_type="audio_chunk",
                )

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"客户端 {client_address} (会话: {session_id}) 正常断开连接。")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"客户端 {client_address} (会话: {session_id}) 连接因错误关闭: {e}")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket 处理器发生错误: {e}", exc_info=True)
    finally:
        logger.info(f"会话 {session_id} 已结束，客户端 {client_address}。")
        if hasattr(chat_engine, 'on_client_disconnect'):
            await chat_engine.on_client_disconnect(session_id)


async def main():
    """主函数，初始化组件并启动 WebSocket 服务器。"""
    server = None  # 在 try 块外部定义 server

    logger.info(f"在 ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT} 上启动 WebSocket 服务器")

    # 将 websockets.serve 的结果赋值给 server 变量
    server = await websockets.serve(websocket_connection_handler, WEBSOCKET_HOST, WEBSOCKET_PORT)
    logger.info("WebSocket 服务器已启动并正在运行。")

    try:
        await initialize_chatbot_components()
    except Exception as e:
        logger.error(f"初始化 chatbot 组件失败: {e}", exc_info=True)
        return

    try:
        await asyncio.Future()  # 永远运行直到被中断
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("服务器接收到关闭信号...")
    finally:
        if server:
            logger.info("正在关闭 WebSocket 服务器...")
            server.close()
            await server.wait_closed()
            logger.info("WebSocket 服务器已关闭。")

        if module_manager:
            logger.info("正在关闭模块...")
            await module_manager.shutdown_modules()

        if chat_engine and hasattr(chat_engine, 'shutdown'):
            logger.info("正在关闭 ChatEngine...")
            await chat_engine.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run() 内部会处理 KeyboardInterrupt 并取消任务
        logger.info("应用程序因 KeyboardInterrupt 正在退出...")
    except Exception as e:
        logger.error(f"应用程序运行失败或意外中断: {e}", exc_info=True)
    finally:
        logger.info("应用程序关闭过程完成。")
