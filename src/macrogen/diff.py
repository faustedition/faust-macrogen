from argparse import ArgumentParser
from difflib import SequenceMatcher
from itertools import zip_longest, combinations
from typing import Union

from .graph import MacrogenesisInfo
from pathlib import Path
from .report import HtmlTable, _fmt_node, SingleItem, write_html


def load_info(path: Union[Path, str]) -> MacrogenesisInfo:
    if not isinstance(path, Path):
        path = Path(path)  # always wanted to write sth like this
    if path.is_dir():
        path /= "macrogen-info.zip"
    return MacrogenesisInfo(path)


def diff_order_table(a: MacrogenesisInfo, b: MacrogenesisInfo, title_a: str = "a", title_b: str = "b"):
    table = (HtmlTable()
             .column(title_a, _fmt_node, attrs={'class': 'pull-right'})
             .column('op', attrs={'class': 'pure-center'})
             .column(title_b, _fmt_node))
    diff = SequenceMatcher(a=a.order, b=b.order)
    for op, i1, i2, j1, j2 in diff.get_opcodes():
        if op == "replace":
            for ref_a, ref_b in zip_longest(a.order[i1:i2], b.order[j1:j2]):
                table.row((ref_a, '↔', ref_b), class_='replace')
        elif op == "delete":
            for ref_a in a.order[i1:i2]:
                table.row((ref_a, '−', ''), class_='delete')
        elif op == "insert":
            for ref_b in b.order[j1:j2]:
                table.row(('', '+', ref_b), class_='insert')
        elif op == "equal":
            table.row(SingleItem(f'{i2-i1} gleiche Referenzen ({a.order[i1]} … {a.order[i2-1]})'), class_='equal pure-center ignore')
    return table


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Compare the order of macrogenesis results")
    parser.add_argument("base", type=Path, help="base path (or zip) for comparison")
    parser.add_argument("compare", nargs="+", type=Path, help="comparison paths (or zips)")
    parser.add_argument("-p", "--pairwise", action="store_true", default=False, help="compare all paths pairwise instead of base to all")
    parser.add_argument("-o", "--output-dir", default=Path(), type=Path)
    return parser

def main():
    options = get_argparser().parse_args()
    pairs = list(combinations([options.base] + options.compare, 2)) if options.pairwise \
            else [(options.base, compare) for compare in options.compare]
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for a, b in pairs:
        table = diff_order_table(load_info(a), load_info(b), a.stem, b.stem)
        write_html(options.output_dir / f"order-{a.stem}.{b.stem}.php", table.format_table(), f"{a.stem} / {b.stem}")

if __name__ == '__main__':
    main()
