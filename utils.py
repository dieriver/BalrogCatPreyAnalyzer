import sys
import os
import logging
from logging import handlers

# We configure the logging
logger = logging.getLogger("cat_logger")

cat_cam_py = os.getenv('CAT_PREY_ANALYZER_PATH')


def init_logger():
    logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler = logging.handlers.RotatingFileHandler(
        filename='/var/log/balrog-logs/cat_logger.log',
        maxBytes=(1024*1024*500),
        backupCount=5,
        encoding='utf-8'
    )
    dbg_file_handler = logging.handlers.RotatingFileHandler(
        filename='/var/log/balrog-logs/cat_logger-dbg.log',
        maxBytes=(1024*1024*500),
        backupCount=5,
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