import re
from pathlib import Path
from typing import List, Optional
import reprlib

from .config import config
from lxml import etree
from os import fspath


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


class Document:

    def __init__(self, source: Path):
        tree: etree._ElementTree = etree.parse(fspath(source))
        self.kind = tree.getroot().tag
        idno_els: List[etree.ElementBase] = tree.findall('//f:idno', config.namespaces)
        self.idnos = {el.get('type'): el.text for el in idno_els}
        self.sigil = self.idnos.get('faustedition')
        tt_el: etree._Element = tree.find('//f:textTranscript', config.namespaces)
        if tt_el is not None:
            self.text_transcript: Path = config.path.data.joinpath(tt_el.base.replace('faust://xml/', ''), tt_el.get('uri'))
            transcript: etree._Element = etree.parse(fspath(self.text_transcript))
            self.inscriptions = transcript.xpath('//tei:change[@type="segment"]/@xml:id', namespaces=config.namespaces)
        else:
            self.text_transcript = None
            self.inscriptions = []

    @property
    def uri(self) -> str:
        return faust_uri(self.sigil)

    @property
    def inscription_uris(self) -> List[str]:
        return [faust_uri(self.sigil, inscription=inscr) for inscr in self.inscriptions]

    @property
    def doc_uris(self) -> List[str]:
        return [faust_uri(sigil, idno_type) for idno_type, sigil in self.idnos.items()]

    @property
    def all_inscription_uris(self) -> List[str]:
        return [faust_uri(sigil, idno_type, inscription) for idno_type, sigil in self.idnos.items()
                                                         for inscription in self.inscriptions]

    def __str__(self):
        return self.sigil

    def __repr__(self):
        return f'<{self.__class__}: {self.sigil}, {len(self.inscriptions)} inscriptions>'

def all_documents(path : Optional[Path] = None):
    if path is None:
        path = config.path.data.joinpath('document')
    return [Document(doc) for doc in config.progress(list(path.rglob('**/*.xml')))]
