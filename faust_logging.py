import logging
from logging.config import dictConfig
from pathlib import Path

import yaml


def setup_logging():
    log_config = Path('logging.yaml')
    if log_config.exists():
        dictConfig(yaml.load(log_config.read_text()))
    else:
        logging.basicConfig(level=logging.WARNING)

setup_logging()