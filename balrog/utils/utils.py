import importlib.resources as resources
import logging
import sys
from contextlib import AbstractContextManager
from logging import handlers
from pathlib import Path

from balrog.config import logging_config

# We declare the logger that we use in this package
logger = logging.getLogger(__name__)


class Logging:
    @staticmethod
    def init_logger(stdout_logging_level: int, max_log_size: int, max_log_files: int) -> None:
        logger.setLevel(logging.DEBUG)

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=f'{logging_config.log_base_folder}/{logging_config.log_file_name}',
            maxBytes=(1024*1024*max_log_size),
            backupCount=max_log_files,
            encoding='utf-8'
        )
        dbg_file_handler = logging.handlers.RotatingFileHandler(
            filename=f'{logging_config.log_base_folder}/{logging_config.log_dbg_file_name}',
            maxBytes=(1024*1024*500),
            backupCount=2,
            encoding='utf-8'
        )
        stdout_handler.setLevel(stdout_logging_level)
        file_handler.setLevel(stdout_logging_level)
        dbg_file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stdout_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        dbg_file_handler.setFormatter(formatter)

        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        logger.addHandler(dbg_file_handler)

    @staticmethod
    def clean_logs() -> list[str]:
        base_path = Path(logging_config.log_base_folder)
        removed_files = []
        for file in base_path.iterdir():
            if file.is_dir() or (file.is_file() and (file.name == logging_config.log_file_name or file.name == logging_config.log_dbg_file_name)):
                continue
            else:
                removed_files.append(str(file))
                file.unlink(missing_ok=True)
        return removed_files


def get_resource_path(resource_name: str) -> AbstractContextManager[Path]:
    return resources.as_file(resources.files("balrog.resources").joinpath(resource_name))
