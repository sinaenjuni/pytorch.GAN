
import logging
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler('./logging_tset.txt')
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(level=logging.DEBUG)

# logging.basicConfig(filename=, level=logging.DEBUG)

logger.debug('my DEBUG log')
logger.info('my INFO log')
logger.warning('my WARNING log')
logger.error('my ERROR log')
logger.critical('my CRITICAL log')