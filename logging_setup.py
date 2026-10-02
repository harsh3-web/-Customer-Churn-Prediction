import logging
import os
from datetime import datetime


def setup_logger(script_name):
    """Return a logger that writes to logs/<script_name>_<YYYYMMDD>.log."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(script_name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)
    log_file = os.path.join("logs", f"{script_name}_{datetime.now():%Y%m%d}.log")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    ))
    logger.addHandler(handler)
    return logger
