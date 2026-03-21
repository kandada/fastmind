"""日志工具"""

import logging
import sys
from typing import Optional

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """获取或创建 logger

    Args:
        name: logger 名称
        level: 日志级别，默认 INFO

    Returns:
        Logger 实例
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level or logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level or logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _loggers[name] = logger
    return logger
