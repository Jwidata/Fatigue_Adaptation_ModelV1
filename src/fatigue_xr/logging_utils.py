from __future__ import annotations

import logging
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    if fields:
        kv = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("%s %s", event, kv)
    else:
        logger.info("%s", event)
