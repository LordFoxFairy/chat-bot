import pytest
import yaml
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

from backend.utils.config_manager import (
    ConfigManager,
    ConfigurationError,
    mask_sensitive_fields,
    unmask_sensitive_fields,
    is_sensitive_field,
    get_config_manager,
    MASK_PLACEHOLDER
)
from backend.core.models.config_data import ConfigData

# Sample config data for testing
SAMPLE_CONFIG = {
    "modules": {
        "llm": {
            "type": "openai",
            "api_key": "sk-123456",
            "model": "gpt-4"
        },
        "tts": {
            "type": "edge-tts"
        }
    },
    "server": {
        "host": "localhost",
        "port": 8000,
        "secret_key": "secret-value"
    }
}

@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config file"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(SAMPLE_CONFIG, f)
    return config_path

@pytest.fixture
def config_manager(config_file):
    """Create a ConfigManager instance"""
    return ConfigManager(str(config_file))

class TestConfigManager:
    """Tests for ConfigManager class"""

    def test_init(self, config_file):
        """Test initialization"""
        manager = ConfigManager(str(config_file))
        assert manager.config_path == config_file
        assert manager._config_cache is None

    @pytest.mark.asyncio
    async def test_get_config_full(self, config_manager):
        """Test getting full configuration"""
        result = await config_manager.get_config(mask_sensitive=False)
        assert isinstance(result, ConfigData)
        assert result.content == SAMPLE_CONFIG
        assert result.section is None
        # Verify cache is populated
        assert config_manager._config_cache == SAMPLE_CONFIG

    @pytest.mark.asyncio
    async def test_get_config_section(self, config_manager):
        """Test getting configuration section"""
        result = await config_manager.get_config(section="server", mask_sensitive=False)
        assert result.section == "server"
        assert result.content == {"server": SAMPLE_CONFIG["server"]}

    @pytest.mark.asyncio
    async def test_get_config_invalid_section(self, config_manager):
        """Test getting invalid section"""
        with pytest.raises(ConfigurationError, match="配置区段 'invalid' 不存在"):
            await config_manager.get_config(section="invalid")

    @pytest.mark.asyncio
    async def test_get_config_mask_sensitive(self, config_manager):
        """Test getting configuration with masked sensitive fields"""
        result = await config_manager.get_config(mask_sensitive=True)
        content = result.content

        # Check masked fields
        assert content["modules"]["llm"]["api_key"] == MASK_PLACEHOLDER
        assert content["server"]["secret_key"] == MASK_PLACEHOLDER

        # Check unmasked fields
        assert content["modules"]["llm"]["model"] == "gpt-4"
        assert content["server"]["host"] == "localhost"

    @pytest.mark.asyncio
    async def test_load_config_file_not_found(self, tmp_path):
        """Test loading non-existent config file"""
        manager = ConfigManager(str(tmp_path / "non_existent.yaml"))
        with pytest.raises(ConfigurationError, match="不存在"):
            await manager.get_config()

    @pytest.mark.asyncio
    async def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML file"""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("key: : value")  # Invalid YAML syntax

        manager = ConfigManager(str(config_path))
        with pytest.raises(ConfigurationError, match="格式错误"):
            await manager.get_config()

    @pytest.mark.asyncio
    async def test_update_config_full(self, config_manager, config_file):
        """Test updating full configuration"""
        updates = {
            "modules": {
                "llm": {"model": "gpt-5"} # Update existing
            },
            "new_section": {"enabled": True} # Add new
        }

        result = await config_manager.update_config(updates)

        # Check runtime result
        assert result.content["modules"]["llm"]["model"] == "gpt-5"
        assert result.content["new_section"]["enabled"] is True

        # Check file persistence
        with open(config_file, "r") as f:
            saved_config = yaml.safe_load(f)
        assert saved_config["modules"]["llm"]["model"] == "gpt-5"
        assert saved_config["new_section"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_update_config_section(self, config_manager, config_file):
        """Test updating specific section"""
        updates_section = {"port": 9000}

        result = await config_manager.update_config(updates_section, section="server")

        assert result.content["server"]["port"] == 9000
        assert result.content["server"]["host"] == "localhost" # Preserved

        # Verify file persistence
        with open(config_file, "r") as f:
            saved_config = yaml.safe_load(f)
        assert saved_config["server"]["port"] == 9000

    @pytest.mark.asyncio
    async def test_update_config_deep_merge(self, config_manager):
        """Test deep merge logic via update_config"""
        original = {
            "a": {
                "b": 1,
                "c": {"d": 2}
            }
        }
        # Manually set cache to test _deep_merge logic via the update flow
        config_manager._config_cache = original

        # Mock save and validate to focus on merge
        with patch.object(config_manager, '_save_config', new_callable=MagicMock) as mock_save:
            f = asyncio.Future()
            f.set_result(None)
            mock_save.return_value = f

            with patch.object(config_manager, '_validate_config'):
                updates = {
                    "a": {
                        "c": {"e": 3}
                    }
                }
                await config_manager.update_config(updates, validate=False)

                # Get the config passed to save
                saved_config = mock_save.call_args[0][0]

                assert saved_config["a"]["b"] == 1
                assert saved_config["a"]["c"]["d"] == 2
                assert saved_config["a"]["c"]["e"] == 3

    @pytest.mark.asyncio
    async def test_update_config_save_failure(self, config_manager):
        """Test save failure"""
        # Mock _save_config to fail
        with patch.object(config_manager, '_save_config', side_effect=Exception("Disk full")):
            with pytest.raises(ConfigurationError, match="更新配置失败"):
                await config_manager.update_config({"server": {"port": 8001}})

    def test_invalidate_cache(self, config_manager):
        """Test cache invalidation"""
        config_manager._config_cache = {}
        config_manager.invalidate_cache()
        assert config_manager._config_cache is None

class TestSensitiveFunctions:
    """Tests for sensitive data handling functions"""

    @pytest.mark.parametrize("field,expected", [
        ("api_key", True),
        ("OPENAI_API_KEY", True),
        ("client_secret", True),
        ("password", True),
        ("db_password_1", True),
        ("auth_token", True),
        ("access_token", True),
        ("private_key_path", True),
        ("credentials", True),
        ("my_credential", True),
        ("username", False),
        ("host", False),
        ("model_name", False),
        ("KEY_ID", False), # "key" pattern might match, check regex carefully
    ])
    def test_is_sensitive_field(self, field, expected):
        """Test sensitive field detection"""
        # Note: regex includes ".*key.*" or ".*api[_-]?key.*"?
        # checking source: r".*api[_-]?key.*", so "KEY_ID" shouldn't match unless it has "api" or is "private key"
        # Wait, check source patterns:
        # r".*api[_-]?key.*"
        # r".*secret.*"
        # r".*password.*"
        # r".*token.*"
        # r".*credential.*"
        # r".*auth.*"
        # r".*private[_-]?key.*"

        # If "KEY_ID" is passed, it doesn't match these patterns.
        # But "my_private_key" should match.
        assert is_sensitive_field(field) == expected

    def test_mask_sensitive_fields(self):
        """Test masking sensitive fields"""
        data = {
            "api_key": "secret",
            "my_password": "password123",
            "public_info": "hello",
            "nested": {
                "auth_token": "token123",
                "normal": "value"
            },
            "credentials": {
                "user": "u",
                "pass": "p"
            },
            "list": [
                {"secret_key": "k1"},
                {"name": "n1"}
            ]
        }

        masked = mask_sensitive_fields(data)

        assert masked["api_key"] == MASK_PLACEHOLDER
        assert masked["my_password"] == MASK_PLACEHOLDER
        assert masked["public_info"] == "hello"
        assert masked["nested"]["auth_token"] == MASK_PLACEHOLDER
        assert masked["nested"]["normal"] == "value"
        # Dictionary value for sensitive key should be masked entirely
        assert masked["credentials"] == MASK_PLACEHOLDER

        assert masked["list"][0]["secret_key"] == MASK_PLACEHOLDER
        assert masked["list"][1]["name"] == "n1"

    def test_unmask_sensitive_fields_no_change(self):
        """Test unmasking when mask is preserved"""
        original = {"api_key": "real-secret", "host": "localhost"}
        masked_input = {"api_key": MASK_PLACEHOLDER, "host": "localhost"}

        result = unmask_sensitive_fields(masked_input, original)
        assert result["api_key"] == "real-secret"
        assert result["host"] == "localhost"

    def test_unmask_sensitive_fields_with_update(self):
        """Test unmasking when user updated value"""
        original = {"api_key": "real-secret", "host": "localhost"}
        masked_input = {"api_key": "new-secret", "host": "localhost"}

        result = unmask_sensitive_fields(masked_input, original)
        assert result["api_key"] == "new-secret"

    def test_unmask_nested(self):
        """Test nested unmasking"""
        original = {"aws": {"secret": "real", "region": "us-east-1"}}
        masked_input = {"aws": {"secret": MASK_PLACEHOLDER, "region": "us-west-1"}}

        # Note: unmask_sensitive_fields handles "aws" key first.
        # "aws" is not sensitive, so it recurses.

        result = unmask_sensitive_fields(masked_input, original)
        assert result["aws"]["secret"] == "real"
        assert result["aws"]["region"] == "us-west-1"

    def test_unmask_sensitive_dict_value(self):
        """Test unmasking a sensitive field that is a dictionary in original"""
        # If 'credentials' is masked in input:
        original = {"credentials": {"u": "admin", "p": "123"}}
        masked_input = {"credentials": MASK_PLACEHOLDER}

        result = unmask_sensitive_fields(masked_input, original)
        assert result["credentials"] == original["credentials"]

def test_get_config_manager_factory(tmp_path):
    """Test get_config_manager factory and singleton behavior"""
    # Reset global instance for testing
    import backend.utils.config_manager as cm
    old_instance = cm._config_manager
    cm._config_manager = None

    try:
        path1 = str(tmp_path / "c1.yaml")
        path2 = str(tmp_path / "c2.yaml")

        # First call
        m1 = get_config_manager(path1)
        assert isinstance(m1, ConfigManager)
        assert str(m1.config_path) == path1

        # Second call - should return same instance
        m2 = get_config_manager(path2)
        assert m2 is m1

        # Test default path (mocking mostly because we rely on code behavior)
        cm._config_manager = None
        m3 = get_config_manager(None)
        assert str(m3.config_path) == "backend/configs/config.yaml"

    finally:
        # Restore Global state
        cm._config_manager = old_instance
