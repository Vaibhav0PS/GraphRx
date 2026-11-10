"""Load Hetionet and translate edge types between names and abbreviations.

The SIF edge file stores edge types as abbreviations such as "CtD". The
metagraph file defines each type as (source kind, relation, target kind,
direction) and gives the abbreviation for every kind and relation, so we
build the abbreviations from it instead of writing them by hand.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import raw_files

DIRECTION_MARK = {"both": "", "forward": ">", "backward": "<"}


@dataclass(frozen=True)
class EdgeType:
    abbrev: str
    source: str
    relation: str
    target: str
    direction: str


class Metagraph:
    def __init__(self, path: Path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        abbrev = data["kind_to_abbrev"]
        self.node_kinds: list[str] = data["metanode_kinds"]
        self.kind_of_abbrev: dict[str, str] = {abbrev[k]: k for k in self.node_kinds}
        self.abbrev_of: dict[tuple[str, str, str], str] = {}
        self.edge_types: dict[str, EdgeType] = {}
        for source, target, relation, direction in data["metaedge_tuples"]:
            code = abbrev[source] + abbrev[relation] + DIRECTION_MARK[direction] + abbrev[target]
            self.abbrev_of[(source, relation, target)] = code
            self.edge_types[code] = EdgeType(code, source, relation, target, direction)

    def abbrev(self, spec: list[str]) -> str:
        key = tuple(spec)
        if key not in self.abbrev_of:
            known = "\n  ".join(" -".join(k[:2]) + "-> " + k[2] for k in sorted(self.abbrev_of))
            sys.exit(f"Edge type {spec} is not in the Hetionet metagraph. Valid types:\n  {known}")
        return self.abbrev_of[key]


def load(cfg: dict, specs: list[list[str]] | None = None, codes: set[str] | None = None):
    """Return (nodes, edges, metagraph).

    Edge types to keep can be given as [source, relation, target] specs and/or
    as metaedge codes such as "CbG". If neither is given, all edges are kept.
    """
    files = raw_files(cfg)
    for path in files.values():
        if not path.exists():
            sys.exit(f"Missing {path}\nRun first:  python src/download_data.py")

    metagraph = Metagraph(files["metagraph"])
    print("Loading Hetionet ...")
    nodes = pd.read_csv(files["nodes"], sep="\t", dtype=str)
    edges = pd.read_csv(files["edges"], sep="\t", dtype=str)

    unknown = set(edges["metaedge"]) - set(metagraph.edge_types)
    if unknown:
        sys.exit(f"Edge file has types missing from the metagraph: {sorted(unknown)}")

    if specs is not None or codes is not None:
        wanted = {metagraph.abbrev(s) for s in specs or []} | set(codes or [])
        edges = edges[edges["metaedge"].isin(wanted)].reset_index(drop=True)
    print(f"  {len(nodes):,} nodes, {len(edges):,} edges kept")
    return nodes, edges, metagraph
