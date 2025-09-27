import logging
from typing import Optional

def get_logger(name: str = "crispdm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        logger.propagate = False
    return logger