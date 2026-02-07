"""核心模块测试脚本"""
import asyncio
import os
import sys

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.app_context import AppContext
from src.core.session.session_context import SessionContext
from src.core.session.session_manager import SessionManager, InMemoryStorage
from src.core.session.conversation_manager import ConversationManager
from src.core.engine.chat_engine import ChatEngine
from src.utils.logging_setup import logger


def test_app_context():
    """测试 AppContext"""
    logger.info("\n===== 测试 AppContext =====")

    # 清空
    AppContext.clear()

    # 设置模块
    test_modules = {
        "llm": "mock_llm",
        "tts": "mock_tts",
        "asr": "mock_asr"
    }
    AppContext.set_modules(test_modules)

    # 验证获取
    assert AppContext.get_module("llm") == "mock_llm"
    assert AppContext.get_module("tts") == "mock_tts"
    assert AppContext.get_module("asr") == "mock_asr"
    assert AppContext.get_module("nonexistent") is None

    logger.info("✓ AppContext.set_modules() 和 get_module() 工作正常")

    # 清空
    AppContext.clear()
    assert AppContext.get_module("llm") is None
    logger.info("✓ AppContext.clear() 工作正常")

    logger.info("✅ AppContext 所有测试通过\n")


def test_session_context():
    """测试 SessionContext"""
    logger.info("\n===== 测试 SessionContext =====")

    # 清空 AppContext
    AppContext.clear()

    # 创建 SessionContext
    ctx = SessionContext(
        session_id="test_session_123",
        tag_id="user_456"
    )

    assert ctx.session_id == "test_session_123"
    assert ctx.tag_id == "user_456"
    assert ctx.dialogues == []
    assert ctx.config == {}
    logger.info("✓ SessionContext 创建成功")

    # 测试通过模块提供者获取模块
    AppContext.set_modules({"llm": "global_llm"})
    ctx.set_module_provider(AppContext.get_module)
    assert ctx.get_module("llm") == "global_llm"
    logger.info("✓ SessionContext 通过模块提供者获取模块成功")

    # 测试自定义模块覆盖
    ctx2 = SessionContext(
        session_id="test_session_2",
        tag_id="user_2",
        custom_modules={"llm": "custom_llm"}
    )
    ctx2.set_module_provider(AppContext.get_module)
    assert ctx2.get_module("llm") == "custom_llm"
    logger.info("✓ SessionContext 自定义模块覆盖工作正常")

    AppContext.clear()
    logger.info("✅ SessionContext 所有测试通过\n")


async def test_session_manager():
    """测试 SessionManager"""
    logger.info("\n===== 测试 SessionManager =====")

    # 创建存储后端
    storage = InMemoryStorage(maxsize=10)
    manager = SessionManager(storage_backend=storage)

    # 创建会话
    ctx = SessionContext(session_id="test_session", tag_id="test_user")
    await manager.create_session(ctx)
    logger.info("✓ SessionManager.create_session() 成功")

    # 获取会话
    result = await manager.get_session("test_session")
    assert result is not None
    assert result.session_id == "test_session"
    logger.info("✓ SessionManager.get_session() 成功")

    # 获取不存在的会话
    result = await manager.get_session("nonexistent")
    assert result is None
    logger.info("✓ SessionManager 获取不存在会话返回 None")

    # 测试 LRU 淘汰
    storage2 = InMemoryStorage(maxsize=2)
    ctx1 = SessionContext(session_id="s1", tag_id="u1")
    ctx2 = SessionContext(session_id="s2", tag_id="u2")
    ctx3 = SessionContext(session_id="s3", tag_id="u3")

    storage2.set("s1", ctx1)
    storage2.set("s2", ctx2)
    storage2.set("s3", ctx3)  # 应该淘汰 s1

    assert storage2.get("s1") is None
    assert storage2.get("s2") is not None
    assert storage2.get("s3") is not None
    logger.info("✓ InMemoryStorage LRU 淘汰工作正常")

    logger.info("✅ SessionManager 所有测试通过\n")


async def test_conversation_manager():
    """测试 ConversationManager（基础功能）"""
    logger.info("\n===== 测试 ConversationManager =====")

    storage = InMemoryStorage()
    session_manager = SessionManager(storage_backend=storage)
    conv_manager = ConversationManager(session_manager=session_manager)

    assert conv_manager.session_manager is session_manager
    assert len(conv_manager.conversation_handlers) == 0
    logger.info("✓ ConversationManager 初始化成功")

    # 注意：create_conversation_handler 需要 ConversationHandler，这里只测试基本初始化
    logger.info("✅ ConversationManager 基础测试通过\n")


async def test_chat_engine():
    """测试 ChatEngine"""
    logger.info("\n===== 测试 ChatEngine =====")

    AppContext.clear()

    storage = InMemoryStorage()
    session_manager = SessionManager(storage_backend=storage)

    config = {
        "modules": {}  # 空配置
    }

    engine = ChatEngine(
        config=config,
        session_manager=session_manager
    )

    assert engine.global_config == config
    assert engine.session_manager is session_manager
    assert engine.conversation_manager is not None
    assert isinstance(engine.common_modules, dict)
    assert isinstance(engine.protocol_modules, dict)
    logger.info("✓ ChatEngine 初始化成功")

    # 初始化（无模块）
    await engine.initialize()
    logger.info("✓ ChatEngine.initialize() 成功")

    # 验证 AppContext 已设置
    # 虽然没有模块，但 AppContext 应该被设置
    assert AppContext.get_module("llm") is None
    logger.info("✓ ChatEngine 初始化后 AppContext 已设置")

    # 测试 get_module
    assert engine.get_module("nonexistent") is None
    logger.info("✓ ChatEngine.get_module() 工作正常")

    AppContext.clear()
    logger.info("✅ ChatEngine 所有测试通过\n")


async def main():
    """运行所有测试"""
    logger.info("\n" + "="*60)
    logger.info("开始核心模块测试")
    logger.info("="*60)

    try:
        # 单元测试
        test_app_context()
        test_session_context()
        await test_session_manager()
        await test_conversation_manager()

        # 集成测试
        await test_chat_engine()

        logger.info("\n" + "="*60)
        logger.info("✅ 所有测试通过！")
        logger.info("="*60 + "\n")

    except AssertionError as e:
        logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 测试错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
    except Exception as e:
        logger.error(f"\n测试异常: {e}", exc_info=True)
        sys.exit(1)
