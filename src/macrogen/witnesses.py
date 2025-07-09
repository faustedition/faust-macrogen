import re
from collections import defaultdict, Counter
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Optional, Union

from .config import config
from lxml import etree
from os import fspath

"""
Ziele:

* Auflisten aller Dokumente mit Inskriptionen im Text, und der Inskriptionen zu jedem Text
* Für jedes Dokument und jede Inskription: Relevante Szenen / Akte
* Für jede Szene: Relevante Dokumente / Inskriptionen

"""

logger = config.getLogger(__name__)


def encode_sigil(sigil: str) -> str:
    """
    Encodes a human-readable sigil for use in an uri.
    """
    sigil = sigil.replace('α', 'alpha')
    sigil = re.sub(r'\s+', ' ', sigil.strip())
    sigil = re.sub('[^A-Za-z0-9.-]', '_', sigil)
    return sigil


def faust_uri(sigil: str, idno_type: Optional[str] = None, inscription: Optional[str] = None):
    """
    Creates a faust:// URI for a document or inscription.

    Args:
        sigil: the document’s sigil or signature
        idno_type: the identifier of the sigil system used for `sigil`
        inscription: optional inscription id
    """
    components = ['faust:/',
                  'document' if inscription is None else 'inscription',
                  'faustedition' if idno_type is None else idno_type,
                  encode_sigil(sigil)]
    if inscription is not None:
        components.append(inscription)
    return '/'.join(components)


def _ids(reference: str):
    return [s.strip('#') for s in reference.split()]


class BaseDocument:
    ...


class Document(BaseDocument):

    _verses = None
    _paralipomena = None

    def __init__(self, source: Path):
        self.source = source
        tree: etree._ElementTree = etree.parse(fspath(source))
        self.kind = tree.getroot().tag
        idno_els: list[etree.ElementBase] = tree.findall('//f:idno', config.namespaces)
        self.idnos = {el.get('type'): el.text for el in idno_els}
        self.sigil = self.idnos['faustedition']
        tt_el: etree._Element = tree.find('//f:textTranscript', config.namespaces)
        if tt_el is not None:
            self.text_transcript: Path = config.path.data.joinpath(tt_el.base.replace('faust://xml/', ''),
                                                                   tt_el.get('uri'))
            transcript: etree._ElementTree = etree.parse(fspath(self.text_transcript))
            self.inscriptions = transcript.xpath('//tei:change[@type="segment"]/@xml:id', namespaces=config.namespaces)
        else:
            self.text_transcript = None
            self.inscriptions = []

    def verses(self, text_transcript: Optional[etree._ElementTree] = None) -> dict[str, list[int]]:
        if not self._verses:
            try:
                if text_transcript is None:
                    if self.text_transcript is None:
                        raise ValueError(f'There is no textual transcript for {self.sigil}, cannot determine verse coverage')
                text_transcript = etree.parse(fspath(self.text_transcript))

                lines = text_transcript.xpath('//tei:l[@n]', namespaces=config.namespaces) + \
                        text_transcript.xpath('//tei:milestone[@unit="reflines"]', namespaces=config.namespaces)
                insc_lines = defaultdict(list)
                for line in lines:
                    precs = line.xpath('preceding::tei:milestone[@unit="stage"]', namespaces=config.namespaces)
                    linenos = [int(n) for n in _ids(line.get('n')) if n.isdigit()]
                    if precs:
                        prec = precs[0]
                        for insc in _ids(prec.get('change')):
                            insc_lines[insc].extend(linenos)
                    else:
                        insc_lines[''].extend(linenos)

        #                 contained = line.xpath('descendant-or-self::*/@change')
        #                 if contained is not None:
        #                     for change in contained:
        #                         for insc in _ids(change):
        #                             insc_lines[insc].extend(linenos)

                self._verses = insc_lines
            except Exception as e:
                logger.error('Failed to read %s: %s', self, e) 
                self._verses = defaultdict(list)
        return self._verses

    def paralipomena(self):
        if self._paralipomena is None:
            self._paralipomena = set()
            if self.text_transcript:
                text_transcript = etree.parse(fspath(self.text_transcript))
                milestones = text_transcript.xpath('//tei:milestone[@unit="paralipomenon"][@n]', namespaces=config.namespaces)
                for milestone in milestones:
                    m = re.match(r'p([^_]+)', milestone.get('n'))
                    if m:
                        self._paralipomena.add(m.group(1))
        return self._paralipomena



    @property
    def uri(self) -> str:
        return faust_uri(self.sigil)

    @property
    def inscription_uris(self) -> list[str]:
        return [faust_uri(self.sigil, inscription=inscr) for inscr in self.inscriptions]

    @property
    def doc_uris(self) -> list[str]:
        return [faust_uri(sigil, idno_type) for idno_type, sigil in self.idnos.items()]

    @property
    def all_inscription_uris(self) -> list[str]:
        return [faust_uri(sigil, idno_type, inscription) for idno_type, sigil in self.idnos.items()
                for inscription in self.inscriptions]

    def __str__(self):
        return self.sigil

    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.sigil}, {len(self.inscriptions)} inscriptions>'

    def to_record(self):
        record = {
            'sigil': self.sigil,
            'sigil_t': encode_sigil(self.sigil),
            'uri': self.uri,
            'other_sigils': {faust_uri(sigil, kind): sigil for kind, sigil in self.idnos.items()},
            'type': self.kind[self.kind.index('}') + 1:],
        }
        vss = [vs for vs in self.verses().values() if vs]
        if vss:
            record['min_verse'] = min(min(vs) for vs in vss)
            record['max_verse'] = max(max(vs) for vs in vss)
        return record



class Scene:

    def __init__(self, element: etree._Element, parent: 'Scene' = None):
        self.parent = parent
        self.n = element.get('n')
        self.title = element.findtext('f:title', namespaces=config.namespaces)
        self.level = element.tag[element.tag.index('}') + 1:]
        self.subscenes = [Scene(el, parent=self) for el in element.xpath('*[@n]')]
        self.first = element.get('first-verse')
        self.last = element.get('last-verse')

        if self.subscenes:
            if self.first is None:
                self.first = self.subscenes[0].first
            if self.last is None:
                self.last = self.subscenes[-1].last

        if self.first:
            self.first = int(self.first)
        if self.last:
            self.last = int(self.last)


class SceneInfo:

    _instance: 'SceneInfo' = None

    @classmethod
    def get(cls) -> 'SceneInfo':
        if cls._instance is None:
            cls._instance = SceneInfo()
        return cls._instance

    def __init__(self, et=None):
        logger.debug('Reading scene info ...')
        if et is None:
            et = config.scenes_xml
        self.toplevel = [Scene(el) for el in et.xpath('/*/*')]
        config.getLogger(__name__).debug('Loading scene info ...')

        def _add_recursive(target: list, items: list, predicate):
            for item in items:
                if predicate(item):
                    target.append(item)
                if item.subscenes:
                    _add_recursive(target, item.subscenes, predicate)

        self.scenes = []
        _add_recursive(self.scenes, self.toplevel, lambda s: s.level == 'scene')


class IntervalsMixin:
    _group_re = re.compile(r"^(1\.0\.\d|\d\.\d).*")

    def is_relevant_for(self, first: int, last: int):
        assert isinstance(first, int) and isinstance(last, int), f"first ({first!r}) and last ({last!r}) must be ints!"
        return any(first <= interval['start'] <= last or
                   first <= interval['end'] <= last or
                   first <= interval['start'] and interval['end'] <= last
                   for interval in self.intervals)

    def _init_relevant_scenes(self):
        relevant_scenes = set()
        for scene in SceneInfo.get().scenes:
            if scene.first is None or scene.last is None:
              config.getLogger(__name__).warning('Scene %s has no limits', scene.n)
            elif self.is_relevant_for(scene.first, scene.last):
                relevant_scenes.add(scene)
        self.relevant_scenes = frozenset(relevant_scenes)
        self.max_scenes = self._reduce_scenes(relevant_scenes)
        self.groups = {self._group_re.sub(r'\1', scene.n) for scene in self.relevant_scenes}
        self.group = Counter(self.groups).most_common(1)[0][0] if self.relevant_scenes else None

    @staticmethod
    def _reduce_scenes(scenes: set[Scene]) -> set[Scene]:
        result = set(scenes)
        while len(result) > 1 and any(scene.parent for scene in result):
            result = {scene.parent if scene.parent else scene for scene in result}
        return result




class InscriptionCoverage(IntervalsMixin):

    def __init__(self, document: BaseDocument, json: dict):
        self.document = document
        self.id = json['id']
        self.uri = faust_uri(document.sigil, inscription=self.id)
        self.intervals = json['intervals']
        self._init_relevant_scenes()

    def covered_lines(self):
        return len(set(chain.from_iterable(range(i['start'], i['end']+1) for i in self.intervals)))


class DocumentCoverage(BaseDocument, IntervalsMixin):

    def __init__(self, json):
        self.sigil = json['sigil']
        self.uri = faust_uri(self.sigil)
        self.intervals = json['intervals']   # or should we only represent the non-inscription intervals?
        self.inscriptions = []
        if 'inscriptions' in json:
            for inscription in json['inscriptions']:
                self.inscriptions.append(InscriptionCoverage(self, inscription))
        self._init_relevant_scenes()


class WitInscrInfo:

    def __init__(self):
        logger.debug('Loading document and witness coverage from bargraph ...')
        bargraph = config.genetic_bar_graph
        self.documents = [DocumentCoverage(doc) for doc in bargraph]
        self.by_scene: dict[Scene, Union[InscriptionCoverage, DocumentCoverage]] = defaultdict(list)
        self.by_uri: dict[str, Union[InscriptionCoverage, DocumentCoverage]] = dict()
        self.by_group: dict[str, Union[InscriptionCoverage, DocumentCoverage]] = defaultdict(list)
        for doc in config.progress(self.documents, desc='Analyzing documents', unit=' docs', transient=True):
            self.by_uri[doc.uri] = doc
            for inscription in doc.inscriptions:
                self.by_uri[inscription.uri] = inscription
                for scene in inscription.relevant_scenes:
                    self.by_scene[scene].append(inscription)
            for scene in doc.relevant_scenes:
                self.by_scene[scene].append(doc)
            self.by_group[doc.group].append(doc)

    _instance: 'WitInscrInfo' = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = WitInscrInfo()
        return cls._instance

    def resolve(self, arg: str, inscription: Optional[str]=None):
        if arg.startswith('faust://'):
            uri = arg
        else:
            uri = faust_uri(arg, inscription=inscription)
        if inscription is not None and uri.startswith('faust://document/'):
            uri = "/".join((uri.replace('faust://document', 'faust://inscription'), inscription))
        return self.by_uri[uri]


@lru_cache
def all_documents(path: Optional[Path] = None):
    if path is None:
        path = config.path.data.joinpath('document')
    logger.debug('Reading inscription info from sources in %s...', path)
    files = list(path.glob('**/*.xml'))
    if not files:
        raise ValueError(f'No XML files found in {path}!')
    return [Document(doc) for doc in files]
