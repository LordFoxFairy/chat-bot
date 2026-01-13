import asyncio
import json
import uuid
from typing import Optional

import websockets
from websockets.client import WebSocketClientProtocol


class WebSocketHandler:
    """WebSocket 客户端处理器（异步版本）

    职责:
    - 异步连接 WebSocket 服务器
    - 发送和接收消息
    - 使用 MAC 地址作为 tag_id，动态 session_id 跟踪
    """

    def __init__(self, uri: str):
        """初始化 WebSocket 处理器

        Args:
            uri: WebSocket 服务器 URI (例如 "ws://localhost:8765")
        """
        self.uri = uri
        self.ws: Optional[WebSocketClientProtocol] = None
        self.is_connected = False

        # 获取设备的 MAC 地址作为 tag_id
        mac_address_int = uuid.getnode()
        self.tag_id = ':'.join(("%012x" % mac_address_int)[i:i + 2] for i in range(0, 12, 2)).upper()

        self.session_id: Optional[str] = None

        # 任务管理
        self.receive_task: Optional[asyncio.Task] = None

        print(f"[客户端] WebSocketHandler 已初始化，URI: {self.uri}, Tag ID: {self.tag_id}")

    async def _handle_message(self, raw_message: str):
        """处理接收到的 WebSocket 消息

        Args:
            raw_message: 接收到的原始消息内容
        """
        try:
            message_data = json.loads(raw_message)
            message_type = message_data.get("type")
            received_tag_id = message_data.get("tag_id")
            received_session_id = message_data.get("session_id")

            if message_type == "session_assignment":
                # 服务器分配 session_id
                if received_tag_id == self.tag_id:
                    if received_session_id and self.session_id != received_session_id:
                        self.session_id = received_session_id
                        print(f"[客户端] 已从服务器获取 Session ID: {self.session_id} (Tag ID: {self.tag_id})")
                    else:
                        print(f"[客户端] 收到 Session ID 确认: {self.session_id} (Tag ID: {self.tag_id})")
                else:
                    print(f"[客户端] 收到非本客户端的 Session ID 分配消息 (Tag ID: {received_tag_id})")

            elif message_type == "message":
                # 普通消息
                content = message_data.get("content", "")
                print(f"[客户端] 收到消息 (Tag: {received_tag_id}, Session: {received_session_id}): {content}")

            else:
                print(f"[客户端] 收到未知类型的消息: {raw_message}")

        except json.JSONDecodeError:
            print(f"[客户端] 收到非 JSON 格式消息: {raw_message}")
        except Exception as e:
            print(f"[客户端] 处理消息时发生错误: {e}")

    async def _receive_loop(self):
        """接收消息循环"""
        try:
            async for message in self.ws:
                await self._handle_message(message)
        except websockets.ConnectionClosed:
            print("[客户端] 连接已关闭")
            self.is_connected = False
        except Exception as e:
            print(f"[客户端] 接收消息时发生错误: {e}")
            self.is_connected = False

    async def _send_register(self):
        """发送注册消息"""
        register_message = {
            "type": "register",
            "tag_id": self.tag_id,
            "last_session_id": self.session_id
        }
        await self.ws.send(json.dumps(register_message))
        print(f"[客户端] 已发送注册消息 (Tag ID: {self.tag_id}, 上次 Session ID: {self.session_id})")

    async def start(self, timeout: float = 10.0):
        """启动 WebSocket 连接

        Args:
            timeout: 等待连接和 session_id 分配的超时时间（秒）
        """
        print("[客户端] 正在启动 WebSocket 连接...")

        try:
            # 连接 WebSocket
            self.ws = await websockets.connect(self.uri)
            self.is_connected = True
            print("[客户端] 连接已打开")

            # 发送注册消息
            await self._send_register()

            # 启动接收循环
            self.receive_task = asyncio.create_task(self._receive_loop())

            # 等待 session_id 分配
            start_time = asyncio.get_event_loop().time()
            while self.session_id is None and (asyncio.get_event_loop().time() - start_time < timeout):
                await asyncio.sleep(0.1)

            if self.session_id:
                print(f"[客户端] WebSocket 连接已建立 (Tag: {self.tag_id}, Session: {self.session_id})")
            else:
                print(f"[客户端] WebSocket 连接成功但未在 {timeout} 秒内获取 Session ID")

        except Exception as e:
            print(f"[客户端] 连接失败: {e}")
            self.is_connected = False

    async def send_message(self, content: str):
        """发送结构化消息

        Args:
            content: 要发送的消息内容
        """
        if not self.ws or not self.is_connected or not self.session_id:
            print("[客户端] 无法发送消息：WebSocket 未连接或 Session ID 未分配")
            return

        message_payload = {
            "type": "message",
            "tag_id": self.tag_id,
            "session_id": self.session_id,
            "content": content
        }

        try:
            await self.ws.send(json.dumps(message_payload))
            print(f"[客户端] 已发送消息 (Tag: {self.tag_id}, Session: {self.session_id}): {content}")
        except Exception as e:
            print(f"[客户端] 发送消息时发生错误: {e}")

    async def close(self):
        """关闭 WebSocket 连接"""
        print("[客户端] 正在关闭 WebSocket 连接...")

        # 取消接收任务
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

        # 关闭连接
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            print("[客户端] WebSocket 连接已关闭")


async def main():
    """异步主函数示例"""
    # 1. 初始化 WebSocketHandler
    WEBSOCKET_SERVER_URI = "ws://localhost:8765"
    client_handler = WebSocketHandler(WEBSOCKET_SERVER_URI)

    # 2. 启动连接
    await client_handler.start()

    print("[客户端] 主协程正在执行其他任务...")
    await asyncio.sleep(2)  # 等待连接建立

    # 3. 发送消息
    if client_handler.is_connected and client_handler.session_id:
        await client_handler.send_message(f"你好，服务器！这是我 (Tag ID: {client_handler.tag_id}) 的第一条消息。")
        await asyncio.sleep(1)
        await client_handler.send_message("服务器，你收到这条消息了吗？期待你的回显。")
        await asyncio.sleep(1)
    else:
        print("[客户端] 连接未成功建立或 Session ID 未分配，无法发送消息。")

    # 4. 测试重连
    print("\n" + "=" * 50)
    print("[客户端] 模拟断开连接并重新连接以测试 Tag ID 识别和新 Session ID 分配...")
    print("=" * 50 + "\n")
    await client_handler.close()
    await asyncio.sleep(2)

    # 5. 重新启动连接
    print(f"[客户端] 尝试重新连接 (Tag ID: {client_handler.tag_id}, 上次 Session ID: {client_handler.session_id})")
    await client_handler.start()

    await asyncio.sleep(2)

    if client_handler.is_connected and client_handler.session_id:
        await client_handler.send_message("我回来了！这是重连后的消息。新的 Session ID 应该已经分配了。")
        await asyncio.sleep(1)
    else:
        print("[客户端] 重连失败或 Session ID 未分配。")

    # 6. 等待更多消息
    print("\n" + "=" * 50)
    print("[客户端] 主协程将等待 5 秒以接收更多消息...")
    print("=" * 50 + "\n")
    await asyncio.sleep(5)

    # 7. 关闭连接
    await client_handler.close()
    print("[客户端] 客户端示例完成。")


if __name__ == "__main__":
    asyncio.run(main())
