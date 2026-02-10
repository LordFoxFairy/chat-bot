"""Protocol Adapter 基础测试"""
import asyncio
import os
import sys

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.utils.logging_setup import logger
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.core.session.conversation_manager import ConversationManager
from backend.adapters.protocols.websocket_protocol_adapter import WebSocketProtocolAdapter


def test_websocket_protocol_initialization():
    """测试 WebSocket Protocol 初始化"""
    logger.info("\n===== 测试 WebSocket Protocol Adapter 初始化 =====")

    try:
        # 创建依赖
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)
        conv_manager = ConversationManager(session_manager=session_manager)

        # 创建 WebSocket Protocol
        config = {
            "host": "127.0.0.1",
            "port": 9999
        }

        protocol = WebSocketProtocolAdapter(
            module_id="test_websocket",
            config=config,
            conversation_manager=conv_manager
        )

        # 验证基本属性
        assert protocol.module_id == "test_websocket"
        assert protocol.host == "127.0.0.1"
        assert protocol.port == 9999
        assert protocol.conversation_manager is conv_manager

        logger.info("✓ WebSocket Protocol 初始化成功")
        logger.info(f"  - host: {protocol.host}")
        logger.info(f"  - port: {protocol.port}")
        logger.info(f"  - module_id: {protocol.module_id}")

        # 验证会话映射字典
        assert isinstance(protocol.tag_to_session, dict)
        assert isinstance(protocol.session_to_connection, dict)
        assert isinstance(protocol.connection_to_session, dict)
        logger.info("✓ 会话映射字典初始化正常")

        logger.info("✅ WebSocket Protocol Adapter 初始化测试通过\n")
        return True

    except Exception as e:
        logger.error(f"Protocol 测试失败: {e}", exc_info=True)
        return False


def test_protocol_session_mapping():
    """测试 Protocol 会话映射功能"""
    logger.info("\n===== 测试 Protocol 会话映射 =====")

    try:
        storage = InMemoryStorage()
        session_manager = SessionManager(storage_backend=storage)
        conv_manager = ConversationManager(session_manager=session_manager)

        protocol = WebSocketProtocolAdapter(
            module_id="test_websocket",
            config={"host": "0.0.0.0", "port": 8765},
            conversation_manager=conv_manager
        )

        # 模拟连接对象
        mock_connection = object()

        # 测试创建会话映射
        session_id = protocol.create_session(mock_connection, tag_id="user_123")

        logger.info(f"✓ 创建会话映射: session_id={session_id}")

        # 验证映射
        assert protocol.get_session_id(mock_connection) == session_id
        assert protocol.get_connection(session_id) == mock_connection
        assert protocol.tag_to_session.get("user_123") == session_id

        logger.info("✓ 会话映射查询正常")

        # 测试移除会话
        protocol.remove_session(session_id)
        assert protocol.get_session_id(mock_connection) is None
        assert protocol.get_connection(session_id) is None
        assert "user_123" not in protocol.tag_to_session

        logger.info("✓ 会话移除正常")

        logger.info("✅ Protocol 会话映射测试通过\n")
        return True

    except Exception as e:
        logger.error(f"会话映射测试失败: {e}", exc_info=True)
        return False


def main():
    """运行 Protocol 测试"""
    logger.info("\n" + "="*60)
    logger.info("开始 Protocol Adapter 测试")
    logger.info("="*60)

    try:
        # 测试初始化
        success1 = test_websocket_protocol_initialization()

        # 测试会话映射
        success2 = test_protocol_session_mapping()

        if success1 and success2:
            logger.info("\n" + "="*60)
            logger.info("✅ Protocol Adapter 所有测试通过")
            logger.info("="*60 + "\n")
        else:
            logger.error("\n❌ Protocol Adapter 测试失败")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"\n测试异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
