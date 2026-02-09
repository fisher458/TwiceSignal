from __future__ import annotations

import logging as py_logging
import sys
from typing import Optional

DEFAULT_LOGGER_NAME = "signaltwice"
DEFAULT_LOG_FORMAT = "[%(levelname)s] %(name)s: %(message)s"


def configure_logging(
    *,
    level: int = py_logging.INFO,
    logger_name: str = DEFAULT_LOGGER_NAME,
    stream: Optional[object] = None,
    fmt: str = DEFAULT_LOG_FORMAT,
    force: bool = False,
) -> py_logging.Logger:
    """Configure stream-only logging for the signaltwice logger.

    This avoids file handlers by default.
    """
    logger = py_logging.getLogger(logger_name)
    logger.setLevel(level)

    if stream is None:
        stream = sys.stdout

    if force:
        logger.handlers.clear()

    if not any(isinstance(handler, py_logging.StreamHandler) for handler in logger.handlers):
        handler = py_logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(py_logging.Formatter(fmt))
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            if isinstance(handler, py_logging.StreamHandler):
                handler.setLevel(level)
                handler.setFormatter(py_logging.Formatter(fmt))

    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> py_logging.Logger:
    if name in (None, "", DEFAULT_LOGGER_NAME):
        return py_logging.getLogger(DEFAULT_LOGGER_NAME)
    if name.startswith(f"{DEFAULT_LOGGER_NAME}."):
        return py_logging.getLogger(name)
    return py_logging.getLogger(f"{DEFAULT_LOGGER_NAME}.{name}")


_base_logger = py_logging.getLogger(DEFAULT_LOGGER_NAME)
if not _base_logger.handlers:
    _base_logger.addHandler(py_logging.NullHandler())
