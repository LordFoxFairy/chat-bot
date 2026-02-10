import pytest
import os
import yaml
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from backend.utils.config_loader import ConfigLoader
from backend.core.models.exceptions import ConfigurationError

# Test ConfigLoader.load_config

@pytest.mark.asyncio
async def test_load_config_success():
    """测试成功加载 YAML 配置文件"""
    config_content = """
    app:
        name: "test_app"
    """
    with patch("aiofiles.open", new_callable=MagicMock) as mock_aio_open:
        # 模拟文件句柄
        mock_file = MagicMock()
        # mock_file.read needs to be awaitable
        mock_file.read = AsyncMock(return_value=config_content)
        mock_file.__aenter__.return_value = mock_file
        mock_file.__aexit__.return_value = None
        mock_aio_open.return_value = mock_file

        config = await ConfigLoader.load_config("dummy_path.yaml")
        assert config == {"app": {"name": "test_app"}}

@pytest.mark.asyncio
async def test_load_config_file_not_found():
    """测试配置文件不存在的情况"""
    with patch("aiofiles.open", side_effect=FileNotFoundError):
        with pytest.raises(ConfigurationError) as excinfo:
            await ConfigLoader.load_config("non_existent.yaml")
        assert "未找到" in str(excinfo.value)

@pytest.mark.asyncio
async def test_load_config_yaml_error():
    """测试 YAML 格式错误的情况"""
    with patch("aiofiles.open", new_callable=MagicMock) as mock_aio_open:
        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=": - invalid yaml")
        mock_file.__aenter__.return_value = mock_file
        mock_file.__aexit__.return_value = None
        mock_aio_open.return_value = mock_file

        with pytest.raises(ConfigurationError) as excinfo:
            await ConfigLoader.load_config("invalid.yaml")
        assert "解析配置文件" in str(excinfo.value)

@pytest.mark.asyncio
async def test_load_config_unknown_error():
    """测试未知错误的情况"""
    with patch("aiofiles.open", side_effect=Exception("Unknown error")):
        with pytest.raises(ConfigurationError) as excinfo:
            await ConfigLoader.load_config("dummy.yaml")
        assert "未知错误" in str(excinfo.value)


# Test ConfigLoader.resolve_env_vars

def test_resolve_env_vars_no_env():
    """测试没有环境变量的配置"""
    config = {"key": "value", "number": 123}
    resolved = ConfigLoader.resolve_env_vars(config)
    assert resolved == config

def test_resolve_env_vars_existing_env():
    """测试存在环境变量的解析"""
    config = {"key": "${TEST_ENV_VAR}"}
    with patch.dict(os.environ, {"TEST_ENV_VAR": "resolved_value"}):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["key"] == "resolved_value"

def test_resolve_env_vars_with_default():
    """测试带默认值的环境变量解析（环境变量未设置）"""
    config = {"key": "${TEST_ENV_VAR:default_value}"}
    # 确保环境变量未设置
    with patch.dict(os.environ, {}, clear=True):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["key"] == "default_value"

def test_resolve_env_vars_with_default_and_env_set():
    """测试带默认值的环境变量解析（环境变量已设置）"""
    config = {"key": "${TEST_ENV_VAR:default_value}"}
    with patch.dict(os.environ, {"TEST_ENV_VAR": "env_value"}):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["key"] == "env_value"

def test_resolve_env_vars_missing_no_default():
    """测试环境变量未设置且无默认值（保持原样）"""
    config = {"key": "${TEST_MISSING_VAR}"}
    with patch.dict(os.environ, {}, clear=True):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["key"] == "${TEST_MISSING_VAR}"

def test_resolve_env_vars_nested_dict():
    """测试嵌套字典中的环境变量解析"""
    config = {
        "section": {
            "key": "${TEST_VAR}"
        }
    }
    with patch.dict(os.environ, {"TEST_VAR": "value"}):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["section"]["key"] == "value"

def test_resolve_env_vars_list():
    """测试列表中的环境变量解析"""
    config = {
        "items": ["plain", "${TEST_VAR}", "${TEST_VAR:default}"]
    }
    with patch.dict(os.environ, {"TEST_VAR": "value"}):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["items"] == ["plain", "value", "value"]

def test_resolve_env_vars_mixed_string():
    """测试字符串中混合环境变量的情况"""
    config = {"url": "http://${HOST}:${PORT}/api"}
    with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["url"] == "http://localhost:8080/api"

# 针对 ConfigLoader.resolve_env_vars 方法中的正则表达式错误修复测试
# 在阅读代码时，正则表达式是 re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
# 这个正则无法正确处理嵌套或复杂的默认值（如果包含}），但对于简单情况是OK的。
# 这里主要测试其现有逻辑。

def test_resolve_env_vars_with_empty_default():
    """测试默认值为空字符串的情况"""
    config = {"key": "${TEST_VAR:}"}
    with patch.dict(os.environ, {}, clear=True):
        resolved = ConfigLoader.resolve_env_vars(config)
        assert resolved["key"] == ""
