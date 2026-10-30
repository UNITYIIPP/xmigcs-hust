import os
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)

try:
    import xlog  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - runtime optional dependency
    xlog = None


class _XLogStub:
    """xlog 未安装时的安全替代，所有调用静默忽略"""
    class Options:
        def __init__(self):
            self.file = type('f', (), {'enable': False, 'logger_id': '', 'sub_dir': '', 'flush_on_write': False})()
            self.console = type('f', (), {'enable': False})()
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def initialize(self, *args, **kwargs): pass


if xlog is None:
    xlog = _XLogStub()


class XMIGCSXLogger:
    # 统一初始化一次 xlog，记住是否已经初始化过以及始化是否成功
    """Small wrapper around the package xlog initialization."""

    _base_name = "xmigcs"
    _configured = False
    _enabled = False
    _logger_id = None
    _sub_dir = None

    @classmethod
    def _resolve_config(
        cls,
        logger_id: str | None = None,
        sub_dir: str | None = None,
    ) -> tuple[str, str]:
        resolved_logger_id = logger_id or os.getenv(
            "XMIGCS_XLOG_LOGGER_ID", cls._base_name
        )
        resolved_sub_dir = sub_dir or os.getenv(
            "XMIGCS_XLOG_SUBDIR", cls._base_name
        )
        return resolved_logger_id, resolved_sub_dir

    @classmethod
    def configure(
        cls,
        logger_id: str | None = None,
        sub_dir: str | None = None,
    ) -> bool:
        logger_id, sub_dir = cls._resolve_config(logger_id, sub_dir)

        if cls._configured:
            if (
                cls._logger_id is not None
                and cls._sub_dir is not None
                and (cls._logger_id, cls._sub_dir) != (logger_id, sub_dir)
            ):
                logger.warning(
                    "[XLOG] already initialized, ignoring new config: "
                    f"requested logger_id={logger_id}, sub_dir={sub_dir}; "
                    f"active logger_id={cls._logger_id}, sub_dir={cls._sub_dir}"
                )
            return cls._enabled

        cls._configured = True

        if xlog is None:
            logger.error("[XLOG] xlog import failed, xlog logging disabled")
            cls._enabled = False
            return False

        try:
            opts = xlog.Options()
            opts.file.enable = True
            opts.file.logger_id = logger_id
            opts.file.sub_dir = sub_dir
            opts.file.flush_on_write = False
            opts.console.enable = False
            xlog.initialize(opts)
            cls._enabled = True
            cls._logger_id = logger_id
            cls._sub_dir = sub_dir
            xlog.info(
                "[XLOG] initialized successfully: "
                f"logger_id={logger_id}, sub_dir={sub_dir}"
            )
        except Exception as exc:
            logger.error(
                "[XLOG] initialization failed: "
                f"logger_id={logger_id}, sub_dir={sub_dir}, error={exc}"
            )
            cls._enabled = False

        return cls._enabled

def configure_xlog(
    logger_id: str | None = None,
    sub_dir: str | None = None,
) -> bool:
    return XMIGCSXLogger.configure(logger_id=logger_id, sub_dir=sub_dir)
