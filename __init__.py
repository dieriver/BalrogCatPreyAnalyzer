import sys
import traceback

from main_loop import SequentialCascadeFeeder
from utils import logger


if __name__ == '__main__':
    utils.init_logger()
    sq_cascade = SequentialCascadeFeeder()
    try:
        sq_cascade.queue_handler()
    except Exception as e:
        print("Something wrong happened... Message:", e)
        logger.exception("Something wrong happened... Restarting", e)
        traceback.print_exc()
        sys.exit(1)
    finally:
        sq_cascade.shutdown()
