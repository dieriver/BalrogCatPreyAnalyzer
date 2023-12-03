import sys
import os
import logging
from pathlib import Path
from logging import handlers

cat_cam_py = os.getenv('CAT_PREY_ANALYZER_PATH')


# We configure the logging
logger = logging.getLogger("cat_logger")
log_base_folder = '/var/log/balrog-logs/' if os.getenv("BALROG_LOG_FOLDER") is None else os.getenv("BALROG_LOG_FOLDER")
log_filename = 'cat_logger.log'
log_dbg_filename = 'cat_logger-dbg.log'


class Logging:
    @staticmethod
    def init_logger(stdout_logging_level: int, max_log_size: int, max_log_files: int) -> None:
        logger.setLevel(logging.DEBUG)

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=f'{log_base_folder}/{log_filename}',
            maxBytes=(1024*1024*max_log_size),
            backupCount=max_log_files,
            encoding='utf-8'
        )
        dbg_file_handler = logging.handlers.RotatingFileHandler(
            filename=f'{log_base_folder}/{log_dbg_filename}',
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
        base_path = Path(log_base_folder)
        removed_files = []
        for file in base_path.iterdir():
            if file.is_dir() or (file.is_file() and (file.name == log_filename or file.name == log_dbg_filename)):
                continue
            else:
                removed_files.append(str(file))
                file.unlink(missing_ok=True)
        return removed_files
