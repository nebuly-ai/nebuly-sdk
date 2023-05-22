from logging import Logger
import logging

nebuly_logger: Logger = logging.getLogger(name=__name__)
nebuly_logger.setLevel(level=logging.INFO)
