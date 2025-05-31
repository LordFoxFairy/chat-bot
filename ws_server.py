import asyncio
import websockets
import logging
import json
import wave
import io
import math
import struct
from typing import Optional  # 导入 Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("simple_websocket_server")

WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8766  # 使用一个新端口以避免与之前的服务器冲突


# --- 生成简单的WAV音频数据 ---
def generate_simple_wav(duration_ms=500, freq=440, sample_rate=16000):
    """生成一个简单的单声道16-bit PCM WAV字节流 (正弦波)"""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    max_amplitude = 32767  # 16-bit的最大振幅

    wav_data = io.BytesIO()
    with wave.open(wav_data, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16-bit (2字节)
        wf.setframerate(sample_rate)

        for i in range(num_samples):
            value = int(max_amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
            # 将整数打包为2字节的小端序二进制数据
            packed_value = struct.pack('<h', value)
            wf.writeframesraw(packed_value)

    return wav_data.getvalue()


PREPARED_WAV_BYTES = generate_simple_wav(duration_ms=1000, freq=440)  # 1秒 A4 音调
logger.info(f"已生成 {len(PREPARED_WAV_BYTES)} 字节的WAV音频数据用于测试。")


# 修正点：将 path 参数设为可选
async def connection_handler(websocket: websockets.WebSocketServerProtocol, path: Optional[str] = None):
    """处理单个WebSocket客户端连接。"""
    client_address = websocket.remote_address
    session_id = f"ws_simple_{websocket.id}"
    # 记录实际接收到的 path 值，如果为 None，说明库确实没有传递它
    logger.info(f"客户端 {client_address} 已连接。会话 ID: {session_id}。请求路径 (Path): {path}")

    try:
        async for message in websocket:
            if isinstance(message, str):
                logger.info(f"[{session_id}] 收到文本消息: {message[:200]}")
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    content = data.get("content")

                    if msg_type == "text_message":
                        response_text = f"服务器收到您的文本: '{content}'"
                        logger.info(f"[{session_id}] 发送文本回复: {response_text}")
                        await websocket.send(json.dumps({"type": "text_response", "content": response_text}))

                        # logger.info(f"[{session_id}] 准备发送预生成的WAV音频...")
                        # await websocket.send(PREPARED_WAV_BYTES)  # 发送二进制音频数据
                        # logger.info(f"[{session_id}] 已发送 {len(PREPARED_WAV_BYTES)} 字节的WAV音频。")
                        # 可以在发送完二进制数据后发送一个文本信号，告知客户端音频结束
                        await websocket.send(json.dumps({"type": "audio_stream_end", "message": "预生成音频发送完毕"}))

                    elif msg_type == "audio_stream_end_signal":  # 客户端发送的音频流结束信号
                        logger.info(f"[{session_id}] 收到客户端音频流结束信号。")
                        await websocket.send(json.dumps({"type": "info", "content": "已收到您的音频流结束信号。"}))
                    else:
                        logger.warning(f"[{session_id}] 未知的文本消息类型: {msg_type}")
                        await websocket.send(json.dumps({"type": "error", "message": f"未知消息类型: {msg_type}"}))

                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] 收到的文本消息非JSON格式: {message[:100]}")
                    await websocket.send(json.dumps({"type": "error", "message": "无效的JSON格式。"}))
                except Exception as e:
                    logger.error(f"[{session_id}] 处理文本消息时出错: {e}", exc_info=True)
                    await websocket.send(json.dumps({"type": "error", "message": "服务器处理文本时出错。"}))

            elif isinstance(message, bytes):
                logger.info(f"[{session_id}] 收到二进制音频数据，长度: {len(message)} 字节")
                # 对于收到的音频，简单回复确认
                response_text = f"服务器收到 {len(message)} 字节的音频数据。"
                await websocket.send(json.dumps({"type": "text_response", "content": response_text}))

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"客户端 {client_address} (会话: {session_id}) 正常断开连接。")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"客户端 {client_address} (会话: {session_id}) 连接因错误关闭: {e}")
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket处理器发生意外错误: {e}", exc_info=True)
    finally:
        logger.info(f"会话 {session_id} 已结束，客户端 {client_address}。")


async def main():
    """主函数，启动WebSocket服务器。"""
    server = None
    logger.info(f"准备在 ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT} 上启动独立的WebSocket服务器...")
    try:
        server = await websockets.serve(
            connection_handler,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT
        )
        logger.info(f"独立的WebSocket服务器已在 ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT} 上启动并正在监听。")
        await asyncio.Future()  # 保持服务器运行直到被中断
    except OSError as e:
        logger.critical(f"启动WebSocket服务器时发生OSError (例如，端口 {WEBSOCKET_PORT} 可能已被占用): {e}",
                        exc_info=True)
    except Exception as e:
        logger.critical(f"启动WebSocket服务器时发生未预料的错误: {e}", exc_info=True)
    finally:
        if server:
            logger.info("正在关闭独立的WebSocket服务器...")
            server.close()
            try:
                # wait_closed() 在较新版本的 websockets 中可用
                if hasattr(server, 'wait_closed'):
                    await server.wait_closed()
                else:  # 兼容旧版本可能没有 wait_closed 的情况
                    await asyncio.sleep(0.1)  # 给一个短暂的关闭时间
            except Exception as e_close:
                logger.warning(f"关闭服务器时出错: {e_close}")
            logger.info("独立的WebSocket服务器已关闭。")
        logger.info("服务器主程序退出。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("服务器通过KeyboardInterrupt强制退出。")
    except Exception as e:
        logger.error(f"应用程序运行失败: {e}", exc_info=True)
    finally:
        logger.info("应用程序关闭。")
