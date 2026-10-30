import logging
import os

class XMIGCSLogger:
    """Small wrapper around the package logger."""

    _base_name = "xmigcs"
    _configured = False

    @classmethod
    def _resolve_level(cls, level: int | None = None) -> int:
        if level is not None:
            return level

        level_name = os.getenv("XMIGCS_LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_name, logging.INFO)

    @classmethod
    def configure(cls, level: int | None = None) -> logging.Logger:
        logger = logging.getLogger(cls._base_name)

        if not cls._configured:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
            logger.propagate = False
            cls._configured = True

        logger.setLevel(cls._resolve_level(level))
        return logger

    @classmethod
    def get_logger(cls, name: str | None = None) -> logging.Logger:
        cls.configure()

        if not name:
            return logging.getLogger(cls._base_name)

        if name == cls._base_name or name.startswith(f"{cls._base_name}."):
            logger_name = name
        else:
            logger_name = f"{cls._base_name}.{name}"

        return logging.getLogger(logger_name)

def configure_logging(level: int | None = None) -> logging.Logger:
    return XMIGCSLogger.configure(level)

def get_logger(name: str | None = None) -> logging.Logger:
    return XMIGCSLogger.get_logger(name)
