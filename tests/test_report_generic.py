from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from macrogen import Reference, Witness
from macrogen.report import HtmlTable, _build_attrs, _fmt_node, _edition_link, _invert_mapping, _flatten


def test_build_attrs():
    attrs = _build_attrs({'data_foo': '1', 'title': 'Hunz & Co.'})
    assert attrs == ' data-foo="1" title="Hunz &amp; Co."'


def test_simple_table():
    table = HtmlTable()
    table.column("Col 1")
    table.column("Col 2")
    assert table.format_table() == ('<table class="pure-table"><thead><th>Col 1</th><th>Col 2</th></thead>'
                                    '<tbody>\n\n</tbody></table>')


def test_header():
    table = (HtmlTable().column("Col 1", data_sortable_type="alpha").column("Col 2"))
    assert table._format_header() == '<table class="pure-table"><thead>' \
                                     '<th data-sortable-type="alpha">Col 1</th>' \
                                     '<th>Col 2</th></thead><tbody>'


@pytest.fixture
def trivial_table():
    trivial_table = HtmlTable()
    trivial_table.column("Test")
    trivial_table.row((1,))
    return trivial_table


def test_trivial_col(trivial_table):
    assert trivial_table._format_column(0, 1) == "<td>1</td>"


def test_trivial_row(trivial_table):
    row = trivial_table._format_row(trivial_table.rows[0])
    assert row == "<tr><td>1</td></tr>"


def test_trivial_body(trivial_table):
    html = trivial_table.format_table()
    assert html == '<table class="pure-table"><thead><th>Test</th></thead><tbody>\n' \
                   '<tr><td>1</td></tr>\n' \
                   '</tbody></table>'


def test_early_fail_wrong_col_attr():
    table = HtmlTable()
    with pytest.raises(Exception):
        table.column('Col 1', attrs={'class', 'test'})  # set instead of dict


def test_body():
    table = (HtmlTable()
             .column('Col 1', attrs={'class': 'test'})
             .column('Col 2'))
    table.row((1, 2), data_row='1')
    table.row((3, 4), data_row='2')
    html = table.format_table()
    assert html == '<table class="pure-table">' \
                   '<thead><th>Col 1</th><th>Col 2</th></thead><tbody>\n' \
                   '<tr data-row="1"><td class="test">1</td><td>2</td></tr>\n' \
                   '<tr data-row="2"><td class="test">3</td><td>4</td></tr>\n' \
                   '</tbody></table>'


def test_format_string():
    table = (HtmlTable().column('Test', format_spec='Val: {:02d}'))
    assert table._format_column(0, 1) == "<td>Val: 01</td>"


def test_format_spec():
    table = HtmlTable().column('', format_spec='02d')
    assert table._format_column(0, 1) == '<td>01</td>'


def test_format_fun():
    table = HtmlTable().column('Test', format_spec=lambda v: 'Hello ' + str(v))
    assert table._format_column(0, "World") == "<td>Hello World</td>"


### no tests for the entire reports


def test_fmt_node_ref():
    node = MagicMock(spec=Reference)
    node.__str__ = Mock(return_value='2 H')
    node.filename = Path('2_H.dot')
    assert _fmt_node(node) == '<a href="2_H">2 H</a>'


def test_fmt_node_date():
    _fmt_node(date(1749, 8, 28)) == '1749-08-28'


def test_edition_link():
    wit = MagicMock(spec=Witness)
    wit.__str__ = Mock(return_value='2 H')
    wit.sigil_t = '2_H'
    html = _edition_link(wit)
    assert html == '<a href="/document?sigil=2_H">2 H</a>'


def _test_invert_mapping():
    assert _invert_mapping({1: 'a', 2: 'b', 3: 'b'}) == {'a': 1, 'b': {2, 3}}


def test_flatten():
    assert _flatten([1, 2, [3, [4, None]]]) == [1, 2, 3, 4]
