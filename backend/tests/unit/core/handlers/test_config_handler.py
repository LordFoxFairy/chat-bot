import pytest
import logging
from unittest.mock import AsyncMock, Mock, patch

from backend.core.handlers.config_handler import ConfigHandler
from backend.core.models.config_data import ConfigData
from backend.utils.config_manager import MASK_PLACEHOLDER

# 模拟配置数据
MOCK_CONFIG_CONTENT = {
    "modules": {
        "llm": {
            "model": "gpt-4",
            "api_key": "sk-secret-key-12345",
            "temperature": 0.7
        },
        "tts": {
            "engine": "edge-tts"
        }
    },
    "server": {
        "host": "localhost",
        "port": 8000
    }
}

MOCK_CONFIG_CONTENT_MASKED = {
    "modules": {
        "llm": {
            "model": "gpt-4",
            "api_key": MASK_PLACEHOLDER,
            "temperature": 0.7
        },
        "tts": {
            "engine": "edge-tts"
        }
    },
    "server": {
        "host": "localhost",
        "port": 8000
    }
}

@pytest.fixture
def mock_config_manager():
    """模拟 ConfigManager"""
    mock_manager = AsyncMock()

    # 设置 get_config 的返回值
    async def side_effect_get_config(section=None, mask_sensitive=True):
        content = MOCK_CONFIG_CONTENT_MASKED if mask_sensitive else MOCK_CONFIG_CONTENT
        if section:
            if section in content:
                sub_content = {section: content[section]}
                return ConfigData(section=section, content=sub_content)
            return ConfigData(section=section, content={})
        return ConfigData(section=None, content=content)

    mock_manager.get_config.side_effect = side_effect_get_config

    # 设置 update_config 的返回值
    async def side_effect_update_config(updates, section=None, validate=True):
        # 简单模拟：返回更新后的内容（实际逻辑在 ConfigManager 中，这里假设成功）
        # 注意：这里我们只模拟返回值，实际合并逻辑在测试用例中验证 config_manager.update_config 的调用参数
        return ConfigData(section=section, content=updates)

    mock_manager.update_config.side_effect = side_effect_update_config

    return mock_manager

@pytest.fixture
def config_handler(mock_config_manager):
    """创建 ConfigHandler 实例，并打桩 get_config_manager"""
    with patch('backend.core.handlers.config_handler.get_config_manager', return_value=mock_config_manager):
        handler = ConfigHandler(config_path="dummy/path/config.yaml")
        yield handler

@pytest.mark.asyncio
async def test_init():
    """1. 初始化测试"""
    handler = ConfigHandler(config_path="custom/path/config.yaml")
    assert handler.config_path == "custom/path/config.yaml"

    # 默认路径
    handler_default = ConfigHandler()
    assert handler_default.config_path == "backend/configs/config.yaml"

@pytest.mark.asyncio
async def test_handle_config_get_full(config_handler, mock_config_manager):
    """2. 获取完整配置"""
    result = await config_handler.handle_config_get()

    assert result == MOCK_CONFIG_CONTENT_MASKED
    mock_config_manager.get_config.assert_called_once_with(
        section=None,
        mask_sensitive=True
    )

@pytest.mark.asyncio
async def test_handle_config_get_section(config_handler, mock_config_manager):
    """3. 获取指定部分配置"""
    section = "modules"
    result = await config_handler.handle_config_get(section=section)

    assert "modules" in result
    assert result["modules"] == MOCK_CONFIG_CONTENT_MASKED["modules"]
    mock_config_manager.get_config.assert_called_once_with(
        section=section,
        mask_sensitive=True
    )

@pytest.mark.asyncio
async def test_handle_config_get_masks_sensitive(config_handler):
    """4. 敏感字段被掩码"""
    result = await config_handler.handle_config_get()

    # 检查 api_key 是否为掩码值
    assert result["modules"]["llm"]["api_key"] == MASK_PLACEHOLDER
    # 检查非敏感字段是否保持原样
    assert result["modules"]["llm"]["model"] == "gpt-4"

@pytest.mark.asyncio
async def test_handle_config_set_success(config_handler, mock_config_manager):
    """5. 成功更新配置"""
    new_config = {
        "server": {
            "host": "0.0.0.0",
            "port": 9000
        }
    }

    # 调用更新
    result = await config_handler.handle_config_set(new_config)

    # 验证 get_config 被调用（用于获取原始配置以还原掩码）
    mock_config_manager.get_config.assert_called_with(
        section=None,
        mask_sensitive=False
    )

    # 验证 update_config 被调用
    # 注意：unmask_sensitive_fields 会在 handler 中调用
    mock_config_manager.update_config.assert_called_once()
    call_args = mock_config_manager.update_config.call_args
    assert call_args.kwargs['section'] is None
    assert call_args.kwargs['validate'] is True
    assert call_args.kwargs['updates'] == new_config

@pytest.mark.asyncio
async def test_handle_config_set_invalid_config(config_handler):
    """6. 无效配置抛出异常"""
    invalid_config = "not a dict"

    with pytest.raises(ValueError, match="Valid configuration dictionary required"):
        await config_handler.handle_config_set(invalid_config)

@pytest.mark.asyncio
async def test_handle_config_set_restores_masked_fields(config_handler, mock_config_manager):
    """7. 更新时还原掩码字段"""
    # 模拟前端传来的配置，其中 api_key 仍是掩码值，但改变了 temperature
    config_update = {
        "modules": {
            "llm": {
                "model": "gpt-4",
                "api_key": MASK_PLACEHOLDER,  # 掩码值，应该被还原
                "temperature": 0.9           # 修改的值
            }
        }
    }

    # 预期传递给 update_config 的数据（api_key 还原为真实值）
    expected_update = {
        "modules": {
            "llm": {
                "model": "gpt-4",
                "api_key": "sk-secret-key-12345", # 原始真实值
                "temperature": 0.9
            }
        }
    }

    await config_handler.handle_config_set(config_update)

    # 验证 update_config 调用参数
    mock_config_manager.update_config.assert_called_once()
    call_args = mock_config_manager.update_config.call_args

    # 检查 api_key 是否被还原
    updates_arg = call_args.kwargs['updates']
    assert updates_arg["modules"]["llm"]["api_key"] == "sk-secret-key-12345"
    assert updates_arg["modules"]["llm"]["temperature"] == 0.9

@pytest.mark.asyncio
async def test_handle_config_set_with_section_in_payload(config_handler, mock_config_manager):
    """测试通过 payload 中的 _section 字段指定 section"""
    new_config = {
        "_section": "server",
        "host": "127.0.0.1"
    }

    expected_update = {
        "host": "127.0.0.1"
    }

    await config_handler.handle_config_set(new_config)

    # 验证 section 参数被正确提取
    mock_config_manager.update_config.assert_called_once()
    call_args = mock_config_manager.update_config.call_args
    assert call_args.kwargs['section'] == "server"
    assert call_args.kwargs['updates'] == expected_update

@pytest.mark.asyncio
async def test_handle_config_exception_handling(config_handler, mock_config_manager):
    """测试异常处理"""
    mock_config_manager.get_config.side_effect = Exception("Config error")

    with pytest.raises(Exception, match="Config error"):
        await config_handler.handle_config_get()

    mock_config_manager.update_config.side_effect = Exception("Update error")
    # 恢复 get_config 正常，让它fail在 update_config
    mock_config_manager.get_config.side_effect = None
    mock_config_manager.get_config.return_value = ConfigData(content=MOCK_CONFIG_CONTENT)

    with pytest.raises(Exception, match="Update error"):
        await config_handler.handle_config_set({"key": "value"})
