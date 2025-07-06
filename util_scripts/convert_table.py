import sys
from datetime import date

import pandas as pd
import faust
from lxml import etree
from lxml.builder import ElementMaker
import re

"""
This script can be used to convert an excel table in a special format to macrogenesis XML files.

TODO move out of macrogenesis package?
"""

F = ElementMaker(namespace=faust.namespaces['f'], nsmap={None: faust.namespaces['f']})

def make_group(sigils, kind='', source='', comment='', notes=''):
    if not kind: kind = 'temp-pre'
    if not source: source = 'faust://self'
    rel = F.relation(F.source(uri=source), name=kind)
    if comment: rel.append(F.comment(comment))
    for sigil in sigils:
        uri = to_uri(sigil)
        rel.append(F.item(uri=uri))
    if notes:
        rel.append(etree.Comment(notes))
    return rel


def to_uri(sigil):
    uri = 'faust://document/faustedition/' + re.sub(r'[^0-9A-Za-z_.]', '_', sigil.replace('α', 'alpha'))
    return uri


def convert_relative(filename, root=F.macrogenesis()):
    df = pd.read_excel(filename, sheet_name=0).fillna('')
    df.columns = ['sigil', 'source', 'comment', 'notes']
    group_row = None
    row_items = []
    for row in df.itertuples():
        if row.sigil == '---':
            if group_row and len(row_items) >= 2:
                root.append(make_group(row_items, kind=group_row.source, comment=group_row.comment, notes=group_row.notes))
            group_row = row
            row_items = []
        else:
            row_items.append(row.sigil)
            if len(row_items) >= 2 and (row.source or row.comment):
                root.append(make_group(row_items, comment=row.comment, source=row.source, notes=row.notes))
                row_items = [row.sigil]
    return root


def to_iso(when):
    if isinstance(when, str):
        match = re.match(r'(\d+).(\d+).(\d+)', when)
        if match:
            return date(int(match.group(3)), int(match.group(2)), int(match.group(1))).isoformat()
        else:
            raise ValueError(when)
    elif isinstance(when, date):
        return when.isoformat()
    else:
        raise TypeError(f'When ({when}) is a {type(when)}, but it should be str or date')


def convert_absolute(filename, root=F.macrogenesis()):
    df = pd.read_excel(filename, sheet_name=1).fillna('')
    print(df.columns)
    df.columns = ['sigil', 'source', 'comment', 'notBefore', 'notAfter', 'from', 'to', 'when', 'notes']
    ## dp benutzt nur when
    for row in df.itertuples():
        if row.sigil != '---':
            try:
                el = F.date(
                        F.source(uri=row.source if row.source else 'faust://self'),
                        when=to_iso(row.when))
                if row.comment:
                    el.append(F.comment(row.comment))
                el.append(F.item(uri=to_uri(row.sigil)))
                if row.notes:
                    root.append(etree.Comment(row.notes))
                root.append(el)
            except:
                root.append(etree.Comment(f'Could not convert {row}'))
    return root


if __name__ == '__main__':
    root = convert_relative(sys.argv[1])
    root = convert_absolute(sys.argv[1], root)
    tree = etree.ElementTree(root)
    tree.write(sys.argv[1] + ".xml", encoding="utf-8", pretty_print=True)
