import logging

from rich.logging import RichHandler

LOGGING_LEVEL = logging.INFO
rich_handler = RichHandler(show_time=False, level=LOGGING_LEVEL)

logging.basicConfig(
    level=LOGGING_LEVEL,  # Sets the root logger's level
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[rich_handler],
)

logger = logging.getLogger(__name__)
