import sys
from loguru import logger

# A simple logger configuration
logger.remove()
logger.add(sys.stderr, level="INFO")
loguru_logger = logger