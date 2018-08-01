#!/usr/bin/env python3
from faust_logging import logging

import sys

import graph
from report import report_components, report_refs, report_missing, report_index, report_conflicts, report_help
from visualize import render_all

logger = logging.getLogger('main')


def _main(argv=sys.argv):
    graphs = graph.macrogenesis_graphs()
    report_help()
    report_refs(graphs)
    report_missing(graphs)
    report_components(graphs)
    report_conflicts(graphs)
    report_index()
    render_all()


if __name__ == '__main__':
#    import requests_cache
#    requests_cache.install_cache(expire_after=86400)
    _main()
