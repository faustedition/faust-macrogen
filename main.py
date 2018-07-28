#!/usr/bin/env python3
from faust_logging import logging

import sys

import graph

logger = logging.getLogger('main')


def _main(argv=sys.argv):
    graphs = graph.workflow()


if __name__ == '__main__':
    import requests_cache

    requests_cache.install_cache(expire_after=86400)
    _main()
