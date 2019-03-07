#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

from macrogen.config import config
from . import graph
from . import report
from .visualize import render_all
from .config import config

def main(argv=sys.argv):
    parser = ArgumentParser(argv[0])
    group = parser.add_argument_group("Configuration options")
    config.prepare_options(group)
    options = parser.parse_args(argv[1:])
    config.config_override = vars(options)

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
    report.report_downloads(graphs)
    render_all()


if __name__ == '__main__':
    main()
