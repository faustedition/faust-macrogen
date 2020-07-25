#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from logging import Logger
from pathlib import Path

from macrogen.config import config
from . import graph
from . import report
from .visualize import render_all
from .config import config

def main(argv=sys.argv):
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', metavar='ZIP', help='Loads graph from zip file and runs the reporting', type=Path)
    parser.add_argument('-o', '--output', metavar='ZIP', help='Saves graph to given zip file', type=Path)
    parser.add_argument('--skip-reports', action='store_true', help='Do not write reports (use with -o)')
    parser.add_argument('--override-bibscore', nargs='+', metavar='uri=value', help='Override bibscore for specific URIs')
    group = parser.add_argument_group("Configuration options")
    config.prepare_options(group)
    options = parser.parse_args(argv[1:])
    config.config_override = vars(options)

    if options.override_bibscore:
        logger: Logger = config.getLogger(__name__)
        for override in options.override_bibscore:
            uri, value_str = override.split('=')
            try:
                value = int(value_str)
                config.bibscores[uri] = value

            except ValueError:
                logger.error('Could not parse score %s for source %s', value_str, uri)

    graphs = graph.MacrogenesisInfo(options.input)

    if options.output:
        graphs.save(options.output)

    if config.order:
        report.write_order_xml(graphs)

    if not options.skip_reports:
        report.generate_reports(graphs)
        render_all()



if __name__ == '__main__':
    main()
