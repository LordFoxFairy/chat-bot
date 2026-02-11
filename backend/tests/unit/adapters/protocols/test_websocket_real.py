import pytest
import asyncio
import traceback
import websockets
import json
import yaml
import logging

from backend.adapters.protocols.websocket_protocol_adapter import WebSocketProtocolAdapter
from backend.core.session.conversation_manager import ConversationManager
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.core.session.session_context import SessionContext
from backend.core.models import StreamEvent, EventType, TextData
from backend.utils.logging_setup import logger

# 配置日志
logging.basicConfig(level=logging.INFO)

class TestWebSocketProtocolReal:
    """协议层真实加载测试"""

    @pytest.fixture
    def config_path(self):
        return "/Users/nako/PycharmProjects/github/thefoxfairy/chat-bot/backend/configs/config.yaml"

    @pytest.fixture
    def real_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    @pytest.fixture(scope="function")
    async def session_manager(self):
        storage = InMemoryStorage()
        manager = SessionManager(storage)
        yield manager
        manager.close()

    @pytest.fixture(scope="function")
    async def conversation_manager(self, session_manager):
        manager = ConversationManager(session_manager)
        yield manager
        await manager.destroy_all_handlers()

    @pytest.fixture(scope="function")
    async def websocket_adapter(self, real_config, conversation_manager):
        """创建真实的 WebSocket 适配器"""
        protocol_config = real_config.get("modules", {}).get("protocols", {}).get("config", {}).get("websocket", {})

        # 确保使用测试端口，避免冲突
        protocol_config["port"] = 18765

        adapter = WebSocketProtocolAdapter(
            module_id="test_websocket",
            config=protocol_config,
            conversation_manager=conversation_manager
        )

        # 初始化
        await adapter.setup()

        yield adapter

        # 清理
        try:
            await adapter.stop()
        except:
            pass

    @pytest.mark.asyncio
    async def test_websocket_server_lifecycle(self, websocket_adapter):
        """测试 WebSocket 服务器的启动和停止"""
        task = asyncio.create_task(websocket_adapter.start())

        # 等待服务器启动
        await asyncio.sleep(0.5)

        assert websocket_adapter.server is not None
        assert websocket_adapter.server.is_serving()

        logger.info(f"WebSocket 服务器已启动: ws://{websocket_adapter.host}:{websocket_adapter.port}")

        # 先取消任务，再停止服务器
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        # 停止服务器
        await websocket_adapter.stop()

        # 验证服务器已停止
        assert not websocket_adapter.server.is_serving()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="此测试需要完整的 conversation handler 依赖，应移至集成测试")
    async def test_websocket_connection(self, websocket_adapter):
        """测试 WebSocket 连接和基本通信"""

        task = asyncio.create_task(websocket_adapter.start())
        await asyncio.sleep(0.5) # 等待启动

        uri = f"ws://{websocket_adapter.host}:{websocket_adapter.port}"

        try:
            async with websockets.connect(uri) as websocket:
                logger.info(f"客户端已连接: {uri}")

                # 1. 模拟注册消息
                tag_id = "test_client_001"
                register_msg = StreamEvent(
                    event_type=EventType.SYSTEM_CLIENT_SESSION_START,
                    tag_id=tag_id
                )

                logger.info("发送注册消息...")
                await websocket.send(register_msg.model_dump_json())

                # 2. 接收响应
                logger.info("等待响应...")
                response_json = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response = StreamEvent.model_validate_json(response_json)

                logger.info(f"收到响应: {response}")

                assert response.event_type == EventType.SYSTEM_SERVER_SESSION_START
                assert response.tag_id == tag_id
                assert response.session_id is not None

                session_id = response.session_id

                # 3. 验证会话已创建
                assert websocket_adapter.get_connection(session_id) is not None

        except Exception as e:
            logger.error(f"测试失败: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # 先取消任务，再停止服务器
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            await websocket_adapter.stop()
