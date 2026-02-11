"""真实服务器完整测试

启动真实服务器，测试完整的客户端交互流程。
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestRealServer:
    """真实服务器测试套件"""

    @pytest.fixture(scope="class")
    def server_process(self):
        """启动真实服务器"""
        # 加载环境变量
        env = os.environ.copy()
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env[key] = value

        env["PYTHONPATH"] = str(PROJECT_ROOT)

        # 启动服务器
        process = subprocess.Popen(
            [sys.executable, "-m", "backend.main", "server"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 等待服务器启动
        time.sleep(15)

        yield process

        # 清理
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    @pytest.mark.asyncio
    async def test_01_websocket_connection(self, server_process):
        """测试1: WebSocket 连接"""
        import websockets

        async with websockets.connect("ws://localhost:8765", close_timeout=2) as ws:
            # websockets 14.x 使用 state 属性
            from websockets.protocol import State
            assert ws.state == State.OPEN, "WebSocket 应该是打开状态"

    @pytest.mark.asyncio
    async def test_02_session_registration(self, server_process):
        """测试2: 会话注册"""
        import websockets

        async with websockets.connect("ws://localhost:8765") as ws:
            await ws.send(json.dumps({
                "event_type": "SYSTEM_CLIENT_SESSION_START",
                "event_data": {},
                "tag_id": "session-test"
            }))

            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)

            assert data.get("event_type") == "SYSTEM_SERVER_SESSION_START"
            assert data.get("session_id") is not None

    @pytest.mark.asyncio
    async def test_03_llm_response(self, server_process):
        """测试3: 文本输入 -> LLM 响应"""
        if not os.getenv("API_KEY"):
            pytest.skip("需要设置 API_KEY 环境变量")

        import websockets

        async with websockets.connect("ws://localhost:8765") as ws:
            # 注册会话
            await ws.send(json.dumps({
                "event_type": "SYSTEM_CLIENT_SESSION_START",
                "event_data": {},
                "tag_id": "llm-test"
            }))
            await asyncio.wait_for(ws.recv(), timeout=5)

            # 发送文本
            await ws.send(json.dumps({
                "event_type": "CLIENT_TEXT_INPUT",
                "event_data": {"text": "说一个字：好", "is_final": True},
                "tag_id": "llm-msg"
            }))

            # 收集响应
            text_responses = []
            for _ in range(15):
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(response)
                    if data.get("event_type") == "SERVER_TEXT_RESPONSE":
                        text = data.get("event_data", {}).get("text", "")
                        text_responses.append(text)
                except asyncio.TimeoutError:
                    break

            full_text = "".join(text_responses)
            assert len(full_text) > 0, "应该收到 LLM 响应文本"

    @pytest.mark.asyncio
    async def test_04_tts_audio_response(self, server_process):
        """测试4: TTS 音频响应"""
        if not os.getenv("API_KEY"):
            pytest.skip("需要设置 API_KEY 环境变量")

        import websockets

        async with websockets.connect("ws://localhost:8765") as ws:
            # 注册会话
            await ws.send(json.dumps({
                "event_type": "SYSTEM_CLIENT_SESSION_START",
                "event_data": {},
                "tag_id": "tts-test"
            }))
            await asyncio.wait_for(ws.recv(), timeout=5)

            # 发送文本
            await ws.send(json.dumps({
                "event_type": "CLIENT_TEXT_INPUT",
                "event_data": {"text": "你好", "is_final": True},
                "tag_id": "tts-msg"
            }))

            # 收集响应
            audio_count = 0
            for _ in range(30):
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=3)
                    data = json.loads(response)
                    if data.get("event_type") == "SERVER_AUDIO_RESPONSE":
                        audio_count += 1
                except asyncio.TimeoutError:
                    break

            assert audio_count > 0, "应该收到 TTS 音频包"

    @pytest.mark.asyncio
    async def test_05_multiple_sessions(self, server_process):
        """测试5: 多会话并发"""
        import websockets

        sessions = []
        for i in range(3):
            ws = await websockets.connect("ws://localhost:8765")
            await ws.send(json.dumps({
                "event_type": "SYSTEM_CLIENT_SESSION_START",
                "event_data": {},
                "tag_id": f"multi-{i}"
            }))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            sessions.append((ws, data.get("session_id")))

        # 验证所有会话都有唯一 ID
        session_ids = [s[1] for s in sessions]
        assert len(set(session_ids)) == 3, "每个会话应该有唯一的 session_id"

        # 关闭连接
        for ws, _ in sessions:
            await ws.close()

    @pytest.mark.asyncio
    async def test_06_error_handling(self, server_process):
        """测试6: 错误处理 - 服务器稳定性"""
        import websockets

        async with websockets.connect("ws://localhost:8765") as ws:
            # 发送无效 JSON
            await ws.send("not json")
            await asyncio.sleep(0.5)

            # 发送未注册会话的消息
            await ws.send(json.dumps({
                "event_type": "CLIENT_TEXT_INPUT",
                "event_data": {"text": "test", "is_final": True},
                "tag_id": "error-test"
            }))
            await asyncio.sleep(0.5)

            # 服务器应该还在运行 - 验证可以正常注册
            await ws.send(json.dumps({
                "event_type": "SYSTEM_CLIENT_SESSION_START",
                "event_data": {},
                "tag_id": "recovery-test"
            }))
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)

            assert data.get("event_type") == "SYSTEM_SERVER_SESSION_START"
