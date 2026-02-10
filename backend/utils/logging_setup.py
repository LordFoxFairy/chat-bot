import logging
import logging.handlers
import sys
import os
import json
from typing import Optional, Dict, Any, Union
from backend.core.config_models import LoggingConfig

# 全局默认 logger
logger = logging.getLogger("chat-bot")

class JsonFormatter(logging.Formatter):
    """
    JSON 格式的日志格式化器，用于结构化日志输出
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }

        # 添加异常信息
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)

        return json.dumps(log_record, ensure_ascii=False)

def setup_logging(config: Optional[Union[LoggingConfig, Dict[str, Any]]] = None) -> logging.Logger:
    """
    根据配置初始化日志系统

    Args:
        config: LoggingConfig 对象或配置字典。如果为 None，使用默认配置。

    Returns:
        配置好的 logger 实例
    """
    # 如果传入的是字典，转换为 LoggingConfig 对象
    if isinstance(config, dict):
        # 兼容旧配置结构，如果 config 包含 logging 键，取其值
        if "logging" in config:
            config = LoggingConfig(**config["logging"])
        else:
            config = LoggingConfig(**config)
    elif config is None:
        config = LoggingConfig()

    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level))

    # 清除现有的 handlers
    # 注意：这可能会影响第三方库的日志配置
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # 设置格式化器
    if config.json_format:
        formatter = JsonFormatter(datefmt=config.date_format)
    else:
        formatter = logging.Formatter(
            fmt=config.format,
            datefmt=config.date_format
        )

    # 添加控制台处理器 (Stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 添加文件处理器 (如果配置了路径)
    if config.file_path:
        # 确保日志目录存在
        log_dir = os.path.dirname(config.file_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError as e:
                # 如果创建目录失败，记录错误但不要崩溃，继续使用控制台日志
                sys.stderr.write(f"Failed to create log directory {log_dir}: {e}\n")
                return logger

        try:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=config.file_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding=config.encoding
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except OSError as e:
            sys.stderr.write(f"Failed to setup log file handler: {e}\n")

    # 更新全局 logger
    current_logger = logging.getLogger("chat-bot")

    current_logger.debug(f"Logging initialized with level={config.level}")
    if config.file_path:
        current_logger.debug(f"Logging to file: {config.file_path}")

    return current_logger
