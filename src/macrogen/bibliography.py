import csv
from collections import defaultdict

from .config import config

_bib_labels = {
    'faust://self': 'Faustedition',
    'faust://model/inscription': 'Inskription von',
    'faust://orphan/adoption': 'Datierungsinhalt für',
    'faust://heuristic': '±6 Monate zu',
    'faust://progress': '(Schreibfortschritt)',
    'faust://model/inscription/inline': 'Inskription während Zeuge',
}


class BiblSource:
    """
    A bibliographic source in a macrogenesis XML file.

    Attributes and properties:
        uri (str): The faust://bibliography URI
        detail (str): Detail string like pages
        weight (int): Score for the source
        filename (str): Representation of the source (w/o detail) that is usable as part of a filename
        citation (str): Short citation
        long_citation (str): Detail string for the citation
    """

    def __init__(self, uri, detail=''):
        """
        Creates a bibliographic source.
        Args:
            uri: should be a faust://bibliography/ URI or one of the special values
            detail: detail string like pages
        """
        self.uri = uri
        if detail is None:
            detail = ''
        self.detail = detail
        self.weight = config.bibscores[uri]

    def __eq__(self, other):
        if isinstance(other, BiblSource):
            return self.uri == other.uri and self.detail == other.detail
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash(self.uri) ^ hash(self.detail)

    def __str__(self):
        result = self.citation
        if self.detail is not None:
            result += '\n' + self.detail
        return result

    @property
    def filename(self):
        """
        A string representation of the source (w/o detail) that is usable as part of a filename.
        """
        if self.uri.startswith('faust://bibliography'):
            return self.uri.replace('faust://bibliography/', '')
        else:
            return self.uri.replace('faust://', '').replace('/', '-')

    @property
    def citation(self):
        """
        String representation of only the citation, w/o detail.

        Example:
            Bohnenkamp 19994
        """
        if self.uri in config.bibliography:
            return config.bibliography[self.uri].citation
        elif self.uri in _bib_labels:
            return _bib_labels[self.uri]
        else:
            return self.filename

    @property
    def long_citation(self):
        if self.uri in config.bibliography:
            return config.bibliography[self.uri].reference
        else:
            return self.citation


def read_scores(scorefile):
    """
    Parses the bibliography score file.

    Returns:
        Map uri -> score

    """
    scores = defaultdict(lambda: 1)
    logger = config.getLogger(__name__)
    r = csv.reader(scorefile, delimiter='\t')
    for row in r:
        try:
            scores[row[0]] = int(row[1])
        except ValueError as e:
            logger.warning('Skipping scorefile row %s: %s', row, e)
    return scores
