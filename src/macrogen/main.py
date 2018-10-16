#!/usr/bin/env python3

from .faust_logging import logging

import sys

from . import graph
from . import report
from .visualize import render_all

logger = logging.getLogger('main')


def _main(argv=sys.argv):
    graphs = graph.macrogenesis_graphs()

    report.write_order_xml(graphs)

    report.report_help()
    report.report_refs(graphs)
    report.report_scenes(graphs)
    report.report_missing(graphs)
    report.report_components(graphs)
    report.report_conflicts(graphs)
    report.report_sources(graphs)
    report.report_index(graphs)
    render_all()


if __name__ == '__main__':
    import requests_cache
    requests_cache.install_cache(expire_after=86400)
    _main()
