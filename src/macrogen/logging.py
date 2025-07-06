import logging
from typing import Dict, Union


class LevelFilter(logging.Filter):

    def __init__(self, levels: dict[str, Union[str, int]], default=logging.WARNING):
        """
        Provides a per-logger level limit.

        Args:
            levels: maps logger names to minimum required levels
            default: minimum default level if no more specific stuff is present.

        Examples:

            The following YAML dictConfig provides a filter 'display' that lets pass WARNING
            messages by default, INFO for macrogen.fes and DEBUG for macrogen.fes.FES_Baharev::

                filters:
                  display:
                    (): macrogen.logging.LevelFilter
                    default: WARNING
                    levels:
                      macrogen.fes: INFO
                      macrogen.fes.FES_Baharev: DEBUG

        """
        self.config = {}
        self.default = logging._checkLevel(default)
        for logger, level in levels.items():
            self.config[logger] = logging._checkLevel(level)

    def filter(self, record: logging.LogRecord):
        effective_limit = self.default
        nameparts = record.name.split('.')
        for i in range(len(nameparts), 1, -1):
            prefix = '.'.join(nameparts[:i])
            if prefix in self.config:
                effective_limit = self.config[prefix]
                break
        return record.levelno >= effective_limit