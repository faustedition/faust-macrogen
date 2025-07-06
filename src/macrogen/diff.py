from argparse import ArgumentParser
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import zip_longest, combinations
from typing import Union, Any
from collections.abc import Mapping

from macrogen import Reference, config
from macrogen.splitgraph import SplitReference, Side
from scipy.stats import spearmanr

from .graph import MacrogenesisInfo
from pathlib import Path
from .report import HtmlTable, _fmt_node, SingleItem, write_html, AssertionTable, nodeformatter
import pandas as pd

logger = config.getLogger(__name__)


def load_info(path: Union[Path, str]) -> MacrogenesisInfo:
    if not isinstance(path, Path):
        path = Path(path)  # always wanted to write sth like this
    if path.is_dir():
        path /= "macrogen-info.zip"
    return MacrogenesisInfo(path)


class DiffSide:

    def __init__(self, path: Union[Path, str]):
        if not isinstance(path, Path):
            path = Path(path)  # always wanted to write sth like this
        if path.is_dir():
            path /= "macrogen-info.zip"
        title = path.stem
        if title == "macrogen-info":
            title = path.parent.stem
        self.info = MacrogenesisInfo(path)
        self.title = title
        self.order = self.normalize_order(self.info.order)

    @staticmethod
    def normalize_order(order: list[Reference]) -> list[Reference]:
        if any(isinstance(ref, SplitReference) for ref in order):
            return [ref.reference for ref in order if ref.side == Side.END]
        return order


def visdiff(left: Any, right: Any) -> str:
    if left != right:
        try:
            dir = '<span style="color:green;">⊕</span>' if left < right else '<span style="color:red">⊖</span>'
        except:
            dir = ' '
        return f'{left}⏵<strong>{right}</strong> {dir}'
    else:
        return f'<span class="ignore">{left}</span>'


def attrdiff(name: str, left: Mapping, right: Mapping) -> str:
    if left is None or right is None:
        return ""
    else:
        left_name = left[name]
        if pd.isna(left_name): left_name = "—"
        right_name = right[name]
        if pd.isna(right_name): right_name = "—"
        return visdiff(left_name, right_name)


class MacrogenDiff:

    def __init__(self, a: Union[Path, str], b: Union[Path, str]):
        self.a = DiffSide(a)
        self.b = DiffSide(b)
        self.matcher = SequenceMatcher(a=self.a.order, b=self.b.order)
        self.title = f"{self.a.title} : {self.b.title}"
        self.filename = f"order-{self.a.title}.{self.b.title}"

    def spearman(self):
        df = pd.DataFrame(dict(a=self.a.info.details.position, b=self.b.info.details.position)).dropna()
        return spearmanr(df)

    def refinfo(self, ref: Reference, left_side: DiffSide, right_side: DiffSide):
        try:
            left, right = left_side.info.details.loc[ref], right_side.info.details.loc[ref]
            return [attrdiff(attr, left, right) for attr in ('max_before_date', 'min_after_date', 'rank')]
        except KeyError:
            return ['', '', ''] # ['KeyError', e]

    def diff_order_table(self) -> HtmlTable:
        table = (HtmlTable()
                 .column('nicht vor', attrs={'class': 'right'})
                 .column('nicht nach', attrs={'class': 'right'})
                 .column('Rang', attrs={'class': 'right'})
                 .column(self.a.title, nodeformatter(self.a.title + '/'), attrs={'class': 'right border-right'})
                 .column(self.b.title, nodeformatter(self.b.title + '/'))
                 .column('nicht vor')
                 .column('nicht nach')
                 .column('Rang')
                 )
        for op, i1, i2, j1, j2 in self.matcher.get_opcodes():
            if op == "replace":
                for ref_a, ref_b in zip_longest(self.a.order[i1:i2], self.b.order[j1:j2]):
                    table.row(self.refinfo(ref_a, self.a, self.b) + [ref_a or '', ref_b or ''] + self.refinfo(ref_b, self.a, self.b), class_='replace')
            elif op == "delete":
                for ref_a in self.a.order[i1:i2]:
                    table.row(self.refinfo(ref_a, self.a, self.b) + [ref_a, '', '', '', ''], class_='delete')
            elif op == "insert":
                for ref_b in self.b.order[j1:j2]:
                    table.row(['', '', '', '', ref_b] + self.refinfo(ref_b, self.a, self.b), class_='insert')
            elif op == "equal":
                table.row(SingleItem(
                        f'{i2 - i1} gleiche Referenzen ({_fmt_node(self.a.order[i1])} … {_fmt_node(self.a.order[i2 - 1])})'),
                        class_='equal pure-center ignore')
        return table

    def conflict_diffs(self):
        def unsplit(node):
            if isinstance(node, SplitReference):
                return node.reference
            else:
                return node
        c_a = {(unsplit(u), unsplit(v)) for u, v, k, attr in self.a.info.conflicts}
        c_b = {(unsplit(u), unsplit(v)) for u, v, k, attr in self.b.info.conflicts}
        only_a = c_a - c_b
        only_b = c_b - c_a
        return only_a, only_b

    def conflict_diffs_html(self):
        result = ""
        for side, conflicts in zip([self.a, self.b], self.conflict_diffs()):
            result += f'<h2>{len(conflicts)} Konflikte nur in {side.title}</h2>'
            table = AssertionTable(prefix=side.title + '/')
            for u, v in conflicts:
                for w, x, k, attr in side.info.find_conflicts(u, v):
                    table.edge(w, x, attr)
            result += table.format_table()
        return result


def get_argparser() -> ArgumentParser:
    parser = ArgumentParser(description="Compare the order of macrogenesis results")
    parser.add_argument("base", type=Path, help="base path (or zip) for comparison")
    parser.add_argument("compare", nargs="+", type=Path, help="comparison paths (or zips)")
    parser.add_argument("-p", "--pairwise", action="store_true", default=False,
                        help="compare all paths pairwise instead of base to all")
    parser.add_argument("-o", "--output-dir", default=Path(), type=Path)
    return parser


def main():
    options = get_argparser().parse_args()
    pairs = list(combinations([options.base] + options.compare, 2)) if options.pairwise \
        else [(options.base, compare) for compare in options.compare]
    options.output_dir.mkdir(parents=True, exist_ok=True)
    summary = (HtmlTable()
               .column("Vergleich", lambda diff: f'<a href="{diff.filename}">{diff.title}</a>')
               .column("Spearman-Korrelation", lambda r: "𝜌 = {:0.5f}; p = {:0.5f}".format(*r))
               .column("Ratio")
               .column("+")
               .column("–")
               .column("↔")
               .column("=")
               .column("Rangänderungen")
               .column("Konflikte"))
    for a, b in config.progress(pairs, unit=" Vergleiche"):
        try:
            logger.info('Comparing %s to %s ...', a, b)
            diff = MacrogenDiff(a, b)
            table = diff.diff_order_table()
            output: Path = options.output_dir / (diff.filename + ".php")
            logger.info("Saving %s ...", output.absolute())
            write_html(output, table.format_table() + diff.conflict_diffs_html(),
                       diff.title)
            opcounts = defaultdict(int)
            for op, i1, i2, j1, j2 in diff.matcher.get_opcodes():
                opcounts[op] +=  max(i2-i1, j2-j1)
            rank_changed = sum((diff.a.info.details['rank'] - diff.b.info.details['rank']).dropna() != 0)
            summary.row((diff, diff.spearman(), diff.matcher.ratio(), opcounts['insert'], opcounts['remove'], opcounts['replace'],
                         opcounts['equal'], rank_changed, len(diff.b.info.conflicts) - len(diff.a.info.conflicts)))
        except FileNotFoundError as e:
            logger.error(e)
    write_html(options.output_dir / "order-diff.php", summary.format_table(), head="Vergleiche")

if __name__ == '__main__':
    main()
