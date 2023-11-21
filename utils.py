import sys
import os
import logging
from pathlib import Path
from logging import handlers

# We configure the logging
logger = logging.getLogger("cat_logger")
log_base_folder = '/var/log/balrog-logs/'
log_filename = 'cat_logger.log'
log_dbg_filename = 'cat_logger-dbg.log'

cat_cam_py = os.getenv('CAT_PREY_ANALYZER_PATH')


def init_logger():
    logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=f'{log_base_folder}/{log_filename}',
        maxBytes=(1024*1024*500),
        backupCount=2,
        encoding='utf-8'
    )
    dbg_file_handler = logging.handlers.RotatingFileHandler(
        filename=f'{log_base_folder}/{log_dbg_filename}',
        maxBytes=(1024*1024*500),
        backupCount=2,
        encoding='utf-8'
    )
    stdout_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    dbg_file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    dbg_file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    logger.addHandler(dbg_file_handler)


def clean_logs():
    base_path = Path(log_base_folder)
    for file in base_path.iterdir():
        if file.is_dir() and (file.name == log_filename or file.name == log_dbg_filename):
            continue
        else:
            file.unlink(missing_ok=True)
