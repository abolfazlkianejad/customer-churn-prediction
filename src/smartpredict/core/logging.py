"""Centralized logging setup shared by training and serving entry points."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from smartpredict.config.settings import Settings, get_settings

_managed_handlers: list[logging.Handler] = []


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a JSON string.

        Args:
            record: Log record to render.

        Returns:
            JSON-encoded log entry.

        """
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_formatter(settings: Settings) -> logging.Formatter:
    """Create the formatter selected in settings.

    Args:
        settings: Active application settings.

    Returns:
        Configured logging formatter.

    """
    if settings.log_format == "json":
        return JsonFormatter()
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def configure_logging(settings: Settings | None = None) -> None:
    """Configure the root logger; repeated calls replace prior handlers.

    Only handlers created by this function are removed, so handlers installed
    by test frameworks or an embedding application remain untouched.

    Args:
        settings: Settings to use. Defaults to the process-wide settings.

    """
    active = settings or get_settings()
    root = logging.getLogger()
    root.setLevel(active.log_level)

    for handler in list(_managed_handlers):
        root.removeHandler(handler)
        handler.close()
    _managed_handlers.clear()

    formatter = _build_formatter(active)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    _managed_handlers.append(stream_handler)

    if active.log_to_file:
        active.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(active.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        _managed_handlers.append(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module or component.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        A standard library logger.

    """
    return logging.getLogger(name)
