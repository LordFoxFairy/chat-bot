import websocket  # 导入 WebSocket 客户端库
import threading  # 导入线程库，用于在后台运行 WebSocket 监听
import time  # 导入时间库，用于模拟一些操作
import json  # 导入 JSON 库，用于消息序列化和反序列化
import uuid  # 导入 UUID 库，用于获取 MAC 地址


class WebSocketHandler:
    """
    一个用于处理 WebSocket 连接的封装类（客户端）。
    它负责连接、发送消息以及监听传入消息，并使用 MAC 地址作为固定的 tag_id 和动态的 session_id 跟踪。
    """

    def __init__(self, uri: str):
        """
        初始化 WebSocket 处理器。

        Args:
            uri (str): WebSocket 服务器的 URI (例如 "ws://localhost:8765")。
        """
        self.uri = uri
        self.ws = None  # WebSocket 连接对象
        self.is_connected = False  # 连接状态标志
        self.thread = None  # 用于运行 WebSocket 监听的线程

        # 获取设备的 MAC 地址作为 tag_id
        # uuid.getnode() 返回一个 48 位整数，通常是 MAC 地址。
        # 注意：在某些环境中 (例如 Docker 容器、VM) 或某些网络配置下，
        # 可能会返回一个随机值或虚拟 MAC 地址，或者需要特定权限。
        # 在生产环境中，可能需要更健壮的 MAC 地址获取方法或手动配置 tag_id。
        mac_address_int = uuid.getnode()
        self.tag_id = ':'.join(("%012x" % mac_address_int)[i:i + 2] for i in range(0, 12, 2)).upper()

        self.session_id = None  # 存储当前活跃的 session ID，每次连接成功后由服务器分配

        print(f"[客户端] WebSocketHandler 已初始化，URI: {self.uri}, 固定 Tag ID (MAC): {self.tag_id}")

    def _on_message(self, ws, raw_message: str):
        """
        当接收到 WebSocket 消息时调用。

        Args:
            ws: WebSocket 实例。
            raw_message (str): 接收到的原始消息内容。
        """
        try:
            message_data = json.loads(raw_message)
            message_type = message_data.get("type")
            received_tag_id = message_data.get("tag_id")
            received_session_id = message_data.get("session_id")

            if message_type == "session_assignment":
                # 这是服务器分配 session_id 的消息
                if received_tag_id == self.tag_id:  # 确保是给本客户端的
                    if received_session_id and self.session_id != received_session_id:
                        self.session_id = received_session_id
                        print(
                            f"[客户端] 已从服务器获取/分配新的 Session ID: {self.session_id} (对应 Tag ID: {self.tag_id})")
                    else:
                        print(f"[客户端] 收到 Session ID 确认: {self.session_id} (对应 Tag ID: {self.tag_id})")
                else:
                    print(f"[客户端] 收到非本客户端的 Session ID 分配消息 (Tag ID: {received_tag_id})")
            elif message_type == "message":
                # 这是普通消息
                content = message_data.get("content", "")
                print(f"[客户端] 收到来自 Tag ID {received_tag_id} (Session {received_session_id}) 的消息: {content}")
                # 你可以在这里添加处理接收到消息的自定义逻辑
            else:
                print(f"[客户端] 收到未知类型的消息: {raw_message}")

        except json.JSONDecodeError:
            print(f"[客户端] 收到非 JSON 格式消息: {raw_message}")
        except Exception as e:
            print(f"[客户端] 处理消息时发生错误: {e}")

    def _on_error(self, ws, error):
        """
        当 WebSocket 发生错误时调用。

        Args:
            ws: WebSocket 实例。
            error: 错误对象。
        """
        print(f"[客户端] 发生错误: {error}")
        self.is_connected = False  # 发生错误时，将连接状态设置为断开

    def _on_close(self, ws, close_status_code: int, close_msg: str):
        """
        当 WebSocket 连接关闭时调用。

        Args:
            ws: WebSocket 实例。
            close_status_code (int): 关闭状态码。
            close_msg (str): 关闭消息。
        """
        print(f"[客户端] 连接已关闭。状态码: {close_status_code}, 消息: {close_msg}")
        self.is_connected = False  # 连接关闭时，将连接状态设置为断开

    def _on_open(self, ws):
        """
        当 WebSocket 连接成功打开时调用。

        Args:
            ws: WebSocket 实例。
        """
        print("[客户端] 连接已打开")
        self.is_connected = True  # 连接成功打开时，将连接状态设置为连接

        # 连接建立后，立即向服务器发送注册消息，包含当前 tag_id 和上次的 session_id (可能为 None)
        register_message = {
            "type": "register",
            "tag_id": self.tag_id,
            "last_session_id": self.session_id  # 发送上次的 session_id，服务器可用于判断是否重连
        }
        self.send_raw_message(json.dumps(register_message))
        print(f"[客户端] 已发送注册消息，携带 Tag ID: {self.tag_id}, 上次 Session ID: {self.session_id}")

    def _run_websocket(self):
        """
        内部方法：在单独的线程中运行 WebSocket 应用。
        """
        self.ws = websocket.WebSocketApp(
            self.uri,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        # run_forever 会阻塞，直到连接关闭
        # ping_interval 和 ping_timeout 用于维持连接的活跃性
        self.ws.run_forever(ping_interval=30, ping_timeout=10)

    def start(self):
        """
        启动 WebSocket 连接。这将在一个单独的线程中运行监听循环，
        因此不会阻塞主线程。
        """
        print("[客户端] 正在启动 WebSocket 连接...")
        self.thread = threading.Thread(target=self._run_websocket)
        self.thread.daemon = True  # 设置为守护线程，主程序退出时它也会退出
        self.thread.start()

        # 等待连接建立和 session_id 分配，或者等待一定时间
        timeout = 10  # 秒
        start_time = time.time()
        # 客户端需要等待连接成功且服务器分配了 session_id
        while not self.is_connected or self.session_id is None and (time.time() - start_time < timeout):
            time.sleep(0.1)  # 短暂休眠，避免忙等待

        if self.is_connected and self.session_id:
            print(f"[客户端] WebSocket 连接已启动并成功建立，Tag ID: {self.tag_id}, 当前 Session ID: {self.session_id}")
        else:
            print(f"[客户端] WebSocket 连接启动失败或超时 ({timeout} 秒)。请确保服务器已运行且已分配 Session ID。")

    def send_raw_message(self, raw_message: str):
        """
        通过 WebSocket 发送原始消息（字符串）。
        内部使用，不包含 tag_id/session_id 逻辑。
        """
        if self.ws and self.is_connected:
            try:
                self.ws.send(raw_message)
            except websocket.WebSocketConnectionClosedException:
                print("[客户端] 发送原始消息失败：WebSocket 连接已关闭。")
            except Exception as e:
                print(f"[客户端] 发送原始消息时发生错误: {e}")
        # else:
        # print("[客户端] 无法发送原始消息：WebSocket 未连接或未启动。") # 避免重复打印

    def send_message(self, content: str):
        """
        通过 WebSocket 发送结构化消息（包含 tag_id 和 session_id）。

        Args:
            content (str): 要发送的消息内容。
        """
        if self.ws and self.is_connected and self.tag_id and self.session_id:
            message_payload = {
                "type": "message",
                "tag_id": self.tag_id,
                "session_id": self.session_id,  # 发送当前活跃的 session_id
                "content": content
            }
            try:
                self.send_raw_message(json.dumps(message_payload))
                print(f"[客户端] 已发送消息 (Tag {self.tag_id}, Session {self.session_id}): {content}")
            except Exception as e:
                print(f"[客户端] 构造或发送结构化消息时发生错误: {e}")
        else:
            print("[客户端] 无法发送消息：WebSocket 未连接、未启动或 Tag ID/Session ID 未分配。")

    def close(self):
        """
        关闭 WebSocket 连接。
        """
        if self.ws:
            print("[客户端] 正在关闭 WebSocket 连接...")
            self.ws.close()
            # 如果是守护线程，主程序退出后它也会退出，这里可以等待线程结束
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1)  # 等待线程结束，最多1秒
            print("[客户端] WebSocket 连接已关闭。")
        else:
            print("[客户端] 没有活动的 WebSocket 连接可关闭。")


# --- 示例使用 ---
if __name__ == "__main__":
    # 客户端将尝试连接到本地的 WebSocket 服务器
    WEBSOCKET_SERVER_URI = "ws://localhost:8765"

    # 1. 初始化 WebSocketHandler。tag_id 将自动获取设备的 MAC 地址。
    client_handler = WebSocketHandler(WEBSOCKET_SERVER_URI)

    # 2. 启动连接。这会在后台开始监听并尝试获取 Session ID。
    client_handler.start()

    print("[客户端] 主线程正在执行其他任务...")
    time.sleep(2)  # 模拟主线程的其他工作，等待连接建立和 Session ID 分配

    # 3. 发送消息
    if client_handler.is_connected and client_handler.session_id:
        client_handler.send_message(f"你好，服务器！这是我 (Tag ID: {client_handler.tag_id}) 的第一条消息。")
        time.sleep(1)
        client_handler.send_message("服务器，你收到这条消息了吗？期待你的回显。")
        time.sleep(1)
    else:
        print("[客户端] 连接未成功建立或 Session ID 未分配，无法发送消息。")

    # 模拟客户端断开后重新连接以测试 Tag ID 识别和新 Session ID 分配
    print("\n" + "=" * 50)
    print("[客户端] 模拟断开连接并重新连接以测试 Tag ID 识别和新 Session ID 分配...")
    print("=" * 50 + "\n")
    client_handler.close()
    time.sleep(2)  # 等待连接完全关闭

    # 重新启动连接，这次会尝试使用之前获得的 MAC 地址作为 tag_id
    print(
        f"[客户端] 尝试重新连接，当前 Tag ID: {client_handler.tag_id}, 上次 Session ID (如果有): {client_handler.session_id}")
    client_handler.start()  # 再次调用 start 会重新连接并发送之前的 tag_id

    time.sleep(2)  # 等待重新连接建立

    if client_handler.is_connected and client_handler.session_id:
        client_handler.send_message("我回来了！这是重连后的消息。新的 Session ID 应该已经分配了。")
        time.sleep(1)
    else:
        print("[客户端] 重连失败或 Session ID 未分配。")

    # 模拟等待更多消息或持续运行
    print("\n" + "=" * 50)
    print("[客户端] 主线程将等待 5 秒以接收更多消息...")
    print("=" * 50 + "\n")
    time.sleep(5)

    # 4. 关闭连接
    client_handler.close()
    print("[客户端] 客户端示例完成。")
