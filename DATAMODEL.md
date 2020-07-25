## The Software

After installing this software, you get essentially a Python library called
`macrogen` and a script with the same name. The script can be used to perform
the analysis and report phases that are used in the Faustedition, the library
involves all the data handling.

The script basically consists of two phases: The actual analysis (i.e. reading
the data, building the graph, marking the edges to remove, and deriving an
ordering from that), and a reporting phase that takes the result of the
analysis and builds all the reports and visualizations that can be seen at
<http://faustedition.net/macrogenesis>. In Python, the first phase can be run
by calling `info = macrogen.MacrogenesisInfo()`, the second by calling
`macrogen.report.generate_reports(info)`. From the command line you can skip the report phase by passing `--skip-reports` and skip the analysis phase by passing in a data dump using `-i`. 


