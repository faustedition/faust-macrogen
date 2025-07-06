#!/usr/bin/env python
"""
Handle URIs and the objects they point to
"""

import csv
import re
from abc import ABCMeta
from collections import defaultdict, Counter
from functools import wraps, total_ordering
from operator import itemgetter
from os.path import commonprefix
from pathlib import Path
from typing import ClassVar, Tuple, Type, TypeVar, cast
from datetime import date

from networkx import is_matching
import pandas as pd
from lxml import etree

from macrogen.witnesses import all_documents
from .config import config

logger = config.getLogger(__name__)

F = TypeVar("F")


def call_recorder(
    function: F | None = None, argument_picker=None
) -> F:  # TODO: type hints when Python >= 3.12
    """
    Decorator that records call / result counts.

    Args:
        argument_picker(Function): function that picks the interesting arguments
    """

    def decorator(fun):
        recorder = Counter()

        @wraps(fun)
        def wrapper(*args, **kwargs):
            relevant_args = args if argument_picker is None else argument_picker(args)
            result = fun(*args, **kwargs)
            recorder.update([(relevant_args, result)])
            return result

        wrapper.recorder = recorder  # type: ignore
        return wrapper

    if callable(function) and argument_picker is None:
        return decorator(function)
    else:
        return decorator  # type: ignore


@total_ordering
class Reference(metaclass=ABCMeta):
    """
    Some kind of datable object in the data.
    """

    uri: str
    rank: int = -1
    index: int = -1
    earliest: date | None = None
    latest: date | None = None

    def __init__(self, uri):
        self.uri = uri

    def __lt__(self, other):
        if isinstance(other, Reference):
            return self.uri < other.uri
        else:
            raise TypeError(f"{type(self)} cannot be compared to {type(other)}")

    @property
    def label(self) -> str:
        """
        A human-readable label for the object
        """
        if self.uri.startswith("faust://inscription"):
            kind, sigil, inscription = self.uri.split("/")[-3:]
            return f"{kind}: {sigil} {inscription}"
        elif self.uri.startswith("faust://document"):
            kind, sigil = self.uri.split("/")[-2:]
            return f"{kind}: {sigil}"
        elif self.uri.startswith("faust://bibliography"):
            bibentry = config.bibliography.get(self.uri)
            if bibentry:
                return bibentry.citation
        return self.uri

    def sort_tuple(self) -> tuple[int, str, int, str]:
        """
        Creates a tuple that can be used as a sort key
        """
        match = re.match("faust://(document|inscription)/(.+?)/(.+?)", self.uri)
        if match:
            return (0, match.group(3), 99999, match.group(2))
        else:
            return (0, self.label, 99999, "")

    def __str__(self):
        return self.label

    def __hash__(self):
        return hash(self.uri)

    def __eq__(self, other) -> bool:
        if isinstance(other, Reference):
            return self.uri == other.uri
        else:
            return False

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.uri)})"

    @property
    def filename(self) -> Path:
        """
        File name for this reference

        Returns:
            Path, with extension `.dot`
        """
        match = re.match(
            r"faust://(inscription|document)/(.*?)/(.*?)(/(.*))?(#\S+?)$", self.uri
        )
        if match:
            result = f"{match.group(2)}.{match.group(3)}"
            if match.group(5):
                result += "." + re.sub(r"\W+", "_", match.group(5))
        else:
            result = re.sub(r"\W+", "_", self.uri)
        return Path(result + ".dot")


class Inscription(Reference):
    """
    A reference that refers to an inscription that is not identical with a witness.

    The `witness` attribute contains a reference to the witness.

    Todo:
        maybe support multiple witnesses?
    """

    witness: "Witness"
    inscription: str
    known_witness: bool
    known_inscription: bool
    status: str

    def __init__(self, witness: "Witness", inscription: str):
        uri = "/".join(
            [
                witness.uri.replace("faust://document/", "faust://inscription/"),
                inscription,
            ]
        )
        super().__init__(uri)
        self.witness = witness
        self.inscription = inscription
        self.known_witness = isinstance(witness, Witness)
        if self.known_witness:
            self.known_inscription = inscription in getattr(witness, "inscriptions", {})
            if self.known_inscription:
                self.status = "(ok)"
            else:
                self.status = "unknown-inscription"
        else:
            self.status = "unknown"

    @property
    def label(self):
        return f"{self.witness.label} {self.inscription}"

    def sort_tuple(self):
        v, s1, n, s2 = super().sort_tuple()
        return (v, s1, n, s2 + " " + self.inscription)

    def sigil_sort_key(self):
        return self.witness.sigil_sort_key() + (self.inscription,)


class UnknownRef(Reference):
    """
    A reference in the data that could not be resolved to any kind of known object.
    """

    def __init__(self, uri):
        super().__init__(uri)
        self.status = "unknown"


class AmbiguousRef(Reference):
    """
    A reference that identifies more than one witness.
    """

    def __init__(self, uri, wits):
        super().__init__(uri)
        self.witnesses = frozenset(wits)
        self.status = "ambiguous: " + ", ".join(
            str(wit) for wit in sorted(self.witnesses)
        )

    def first(self) -> "Witness":
        return sorted(self.witnesses, key=str)[0]

    def __add__(self, other):
        new_witnesses = {other} if isinstance(other, Witness) else set(other.witnesses)
        new_witnesses = new_witnesses.union(self.witnesses)
        return AmbiguousRef(self.uri, new_witnesses)

    @property
    def label(self):
        prefix = commonprefix([wit.label for wit in self.witnesses])
        return f"{prefix} … ({len(self.witnesses)})"

    def sort_tuple(self):
        return self.first().sort_tuple()


class Witness(Reference):
    """
    A reference that maps 1:1 to a document in the edition.

    Todo:
        move the database stuff out of here
    """

    database: ClassVar[dict[str, Reference]] = {}
    paralipomena: ClassVar[dict[str, dict]] = {}
    corrections: ClassVar[dict[str, str]] = {}
    sigil: str
    sigil_t: str
    other_sigils: dict[str, str] = {}

    def __init__(self, doc_record):
        # self.sigil_t = None
        if isinstance(doc_record, dict):
            super().__init__(doc_record.get("uri", "?"))
            self.__dict__.update(doc_record)
            self.status = "(ok)"
        else:
            raise TypeError(
                "doc_record must be a mapping, not a " + str(type(doc_record))
            )

    def uris(self):
        """
        Returns a set of all uris that can point to the current witness (i.e., faust://…)
        """
        result = {self.uri}
        for uri in self.other_sigils:
            if self.other_sigils[uri] in {"none", "n.s.", ""}:  # type: ignore
                continue
            uri = uri.replace("-", "_")
            result.add(uri)
            if "/wa_faust/" in uri:
                result.add(uri.replace("/wa_faust/", "/wa/"))
        if getattr(self, "type", "") == "print":
            result.update(
                [uri.replace("faust://document/", "faust://print/") for uri in result]
            )

        result.update({re.sub("[._]{2,}", "_", uri) for uri in result})
        return result

    @classmethod
    def _load_database(cls):
        cls.corrections = cls._load_corrections()
        documents = all_documents()
        sigil_data = [doc.to_record() for doc in documents]
        cls.database = cls.build_database(sigil_data)

    @classmethod
    def _load_corrections(cls):
        result = config.uri_corrections
        logger.info("Loaded %d corrections", len(result))
        return result

    @classmethod
    def _load_paralipomena(cls, url=None):
        if cls.paralipomena is None:
            if url is not None:
                logger.warning(
                    "The url parameter, %s, is deprecated and will be ignored", url
                )
            # cls.paralipomena = config.paralipomena
            cls.paralipomena = {}
            for doc in all_documents():
                for para in doc.paralipomena():
                    cls.paralipomena[para] = doc.to_record()

        return cls.paralipomena

    @classmethod
    def build_database(cls, sigil_data):
        database = {}
        if len(sigil_data) == 0:
            raise ValueError("No sigil data!")
        for doc in sigil_data:
            wit = cls(doc)
            for uri in wit.uris():
                if uri in database:
                    old_entry = database[uri]
                    logger.debug(
                        "URI %s is already in db for %s, adding %s", uri, old_entry, wit
                    )
                    if isinstance(old_entry, AmbiguousRef):
                        database[uri] = old_entry + wit
                    else:
                        database[uri] = AmbiguousRef(uri, {database[uri], wit})
                else:
                    database[uri] = wit
        return database

    @classmethod  # type: ignore   # call_recorder is too complex for typing in Python 3.9
    @call_recorder(argument_picker=itemgetter(1))
    def get(cls: type["Witness"], uri, allow_duplicate=True):
        """
        Returns the reference for the given URI.

        This is the heart of the URI resolving mechanism and the general way to resolve an URI. It will return a
        Reference object representing the concrete item referred to by the URI.

        Args:
            uri: An uri
            allow_duplicate: if an URI resolves to more than one object and allow_duplicate is True, this will resolve
                             to an AmbiguousRef representing all objects that could be referred to by this URI.
                             Otherwise it will return the first matching object.
        """
        if not cls.database:
            cls._load_database()
            cls._load_paralipomena()

        if "#" in uri:
            uri = uri[: uri.index("#")]
        uri = uri.replace("-", "_")

        if uri in cls.corrections:
            correction = cls.corrections[uri]
            if correction in cls.database:
                logger.info(
                    "Corrected %s to %s -> %s",
                    uri,
                    correction,
                    cls.database[correction],
                )
            else:
                logger.warning(
                    "Corrected %s to %s, but it is not in the database", uri, correction
                )
            uri = correction

        if uri in cls.database:
            result = cls.database[uri]
            if isinstance(result, AmbiguousRef):
                return result if allow_duplicate else result.first()
            else:
                return result

        wa_h = re.match(
            r"faust://(inscription|document)/(wa|wa_faust)/2_(I|II|III|IV|V)_H(/(.+))?$",
            uri,
        )
        if wa_h is not None:
            wit = cls.get("faust://document/faustedition/2_H")
            if wa_h.group(4):
                return Inscription(wit, wa_h.group(5))
            else:
                return Inscription(wit, wa_h.group(3))

        wa_pseudo_inscr = re.match(
            r"faust://(inscription|document)/wa/(\S+?)\.?alpha$", uri
        )
        if wa_pseudo_inscr is not None:
            docuri = "faust://document/wa_faust/" + wa_pseudo_inscr.group(2)
            wit = cls.get(docuri)
            if isinstance(wit, Witness):
                return Inscription(wit, "alpha")
            else:
                logger.warning(
                    "Could not fix WA pseudo inscription candidate %s (%s)", uri, wit
                )

        space_inscr = re.match(
            r"faust://(inscription|document)/(.*?)/(.*?)\s+(.*?)", uri
        )
        if space_inscr is not None:
            uri = (
                "faust://inscription/"
                + space_inscr.group(2)
                + "/"
                + space_inscr.group(3)
                + "/"
                + space_inscr.group(4)
            )

        wa_para = re.match(r"faust://(inscription|document)/wa/P(.+?)(/(.+?))$", uri)
        if wa_para and wa_para.group(2) in cls.paralipomena:
            sigil = cls.paralipomena[wa_para.group(2)]["sigil"]
            para_n = wa_para.group(2)
            inscription = wa_para.group(4) if wa_para.group(4) else ("P" + para_n)
            witness = [
                witness
                for witness in list(cls.database.values())
                if isinstance(witness, Witness) and witness.sigil == sigil
            ][0]
            result = Inscription(witness, inscription)
            logger.debug("Recognized WA paralipomenon: %s -> %s", uri, result)
            return result

        if uri.startswith("faust://inscription"):
            match = re.match("faust://inscription/(.*)/(.*)/(.*)", uri)
            if match is not None:
                system, sigil, inscription = match.groups()
                base = "/".join(["faust://document", system, sigil])
                wit = cls.get(base)
                return Inscription(wit, inscription)

        logger.debug("Unknown reference: %s", uri)
        return UnknownRef(uri)

    @property
    def label(self):
        return self.sigil

    def sort_tuple(self):
        p, n, s = self.sigil_sort_key()
        v = 0
        if hasattr(self, "first_verse"):
            v = self.first_verse  # # pyright: ignore
        return v, p, n, s

    def sigil_sort_key(self) -> tuple[str, int, str]:
        """
        A sigil like 2 III H.159:1 currently consists of three parts, basically:

        1. the part before the index number, here "2 III H.", sorted alphabetically
        2. the index number, here "159", sorted numerically
        3. the part after the index number, here ":1", sorted alphabetically.

        This method converts this sigil's
        """
        match = re.match(r"^([12]?\s*[IV]{0,3}\s*[^0-9]+)(\d*)(.*)$", self.sigil)
        if match is None:
            logger.warning("Failed to split sigil %s", self.sigil)
            return (self.sigil, 99999, "")
        split = list(match.groups())

        if split[0] == "H P":  # Paraliponemon
            split[0] = "3 H P"
        if split[1] == "":  # 2 H
            split[1] = -1
        elif split[1].isdigit():
            split[1] = int(split[1])
        else:
            split[2] = split[1] + split[2]
            split[1] = -1

        return tuple(split)  # type: ignore

    @property
    def filename(self):
        return Path(self.sigil_t + ".dot")


### The rest of this module is intended for producing tables about uri use in order to help with disambiguation


def _collect_wits():
    items = defaultdict(list)  # type: dict[Witness | Inscription | UnknownRef, list[tuple[str, int]]]
    macrogenesis_root = config.path.data / "macrogenesis"
    macrogenesis_files = list(macrogenesis_root.glob("**/*.xml"))
    for macrogenetic_file in macrogenesis_files:
        tree = etree.parse(macrogenetic_file)  # type: etree._ElementTree
        for element in cast(
            list[etree.ElementBase],
            tree.xpath("//f:item", namespaces=config.namespaces),
        ):
            uri = element.get("uri")
            wit = Witness.get(uri, allow_duplicate=True)
            items[wit].append(
                (
                    str(macrogenetic_file.relative_to(macrogenesis_root)),
                    element.sourceline,
                )
            )
    logger.info(
        "Collected %d references in %d macrogenesis files",
        len(items),
        len(macrogenesis_files),
    )
    return items


def _assemble_report(wits):
    referenced = [
        (
            wit.uri,
            Witness.corrections.get(wit.uri, ""),
            wit.status,
            ", ".join([file + ":" + str(line) for file, line in wits[wit]]),
        )
        for wit in sorted(wits, key=lambda wit: wit.uri)
    ]

    unused = [
        (str(wit), "", "no macrogenesis data", "")
        for wit in sorted(set(Witness.database.values()), key=str)
        if wit not in wits and isinstance(wit, Witness)
    ]

    return (
        [
            row
            for row in sorted(referenced, key=lambda r: (r[1], r[0]))
            if row[1] != "(ok)"
        ]
        + [row for row in sorted(referenced, key=itemgetter(0)) if row[1] == "(ok)"]
        + unused
    )


def _report_wits(wits, output_csv="witness-usage.csv"):
    with open(output_csv, "w", encoding="utf-8") as reportfile:
        table = csv.writer(reportfile)
        rows = _assemble_report(wits)
        table.writerow(
            ("Zeuge / Referenz", "korrigierte URI (oder leer)", "Status", "Vorkommen")
        )
        for row in rows:
            table.writerow(row)
        stats = Counter([row[1].split(":")[0] for row in rows])
        report = "\n".join(
            "%5d: %s" % (count, status) for status, count in stats.most_common()
        )
        logger.info("Analyzed references in data:\n%s", report)


def _witness_report():
    #    logging.basicConfig(level=logger.WARNING)
    #    logger.setLevel(logging.INFO)
    wits = _collect_wits()

    resolutions = defaultdict(set)
    for uri, result in list(Witness.get.recorder.keys()):  # pyright: ignore
        resolutions[result].add(uri)

    with open("reference-normalizations.csv", "w", encoding="utf-8") as resfile:
        table = csv.writer(resfile)
        table.writerow(("in macrogenesis", "normalisiert"))

        for uri in sorted(resolutions):
            for ref in sorted(resolutions[uri]):
                if str(uri) != str(ref):
                    table.writerow((ref, uri))

    _report_wits(wits)

    wit_sigils = dict()
    for w in [w for w in list(Witness.database.values()) if isinstance(w, Witness)]:
        sigils = dict()
        for uri, sigil in list(w.other_sigils.items()):
            parts = uri.split("/")
            type_, ascii = parts[3:5]
            sigils[type_] = sigil
            sigils[type_ + " (norm.)"] = ascii
        wit_sigils[w] = sigils

    wit_report = pd.DataFrame(wit_sigils).T
    wit_report.to_excel("witness-sigils.xlsx")


if __name__ == "__main__":
    _witness_report()
