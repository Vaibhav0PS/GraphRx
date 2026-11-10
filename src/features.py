"""Metapath features for drug-disease pairs.

A metapath is a pattern of edge types, written with Hetionet abbreviations.
For example "CbGaD" means Compound -binds- Gene -associates- Disease. For
every (drug, disease) pair we compute a degree-weighted count of the walks
that follow the pattern (DWPC, as in Project Rephetio). Hubs, meaning nodes
with many edges, are down-weighted by `damping`, so a path through a gene
that binds 500 drugs counts for less than a path through a gene that binds 2.

Walks that step out and straight back to the same node (X -> Y -> X, e.g. a
gene -> pathway -> same gene) are removed, so counts follow real paths.

Leakage control: the label edge (e.g. Compound-treats-Disease) is rebuilt
from the given training pairs only. A metapath may use the label edge only as
its first or last hop, and walks that return to the starting drug (or end
disease) before using it are removed, so a pair can never reach itself
through its own label edge. train.py additionally gives training positives
out-of-fold features, so their own edge is absent from their graph entirely.

All features are dense (n_drugs x n_diseases) arrays, so features for any
pair are a lookup and scoring every pair is cheap.
"""

import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from hetionet import Metagraph

TOKEN = re.compile(r"[A-Z]+|[a-z]+[<>]?")
FLIP_DIRECTION = str.maketrans("<>", "><")


@dataclass(frozen=True)
class Hop:
    code: str        # metaedge code as stored in the edge file
    transpose: bool  # walk the stored edge backwards
    source: str
    target: str


def parse_metapath(text: str, metagraph: Metagraph) -> list[Hop]:
    tokens = TOKEN.findall(text)
    if "".join(tokens) != text or len(tokens) < 3 or len(tokens) % 2 == 0:
        sys.exit(f"Cannot parse metapath '{text}'. Use Hetionet abbreviations, e.g. CbGaD.")
    kinds, relations = tokens[0::2], tokens[1::2]
    for k in kinds:
        if k not in metagraph.kind_of_abbrev:
            sys.exit(f"Metapath '{text}': unknown node abbreviation '{k}'. Valid: {sorted(metagraph.kind_of_abbrev)}")

    hops = []
    for a, rel, b in zip(kinds, relations, kinds[1:]):
        forward = a + rel + b
        backward = b + rel.translate(FLIP_DIRECTION) + a
        src, tgt = metagraph.kind_of_abbrev[a], metagraph.kind_of_abbrev[b]
        if forward in metagraph.edge_types:
            hops.append(Hop(forward, False, src, tgt))
        elif backward in metagraph.edge_types:
            hops.append(Hop(backward, True, src, tgt))
        else:
            sys.exit(f"Metapath '{text}': no edge type '{a}{rel}{b}' in the metagraph.")
    return hops


class Graph:
    """Sparse 0/1 adjacency matrix per edge type, with node indexes per kind."""

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, metagraph: Metagraph):
        self.metagraph = metagraph
        self.ids: dict[str, np.ndarray] = {}
        self.index: dict[str, dict[str, int]] = {}
        for kind, group in nodes.groupby("kind"):
            ids = group["id"].to_numpy()
            self.ids[kind] = ids
            self.index[kind] = {x: i for i, x in enumerate(ids)}
        self.matrices: dict[str, sp.csr_matrix] = {}
        for code, group in edges.groupby("metaedge"):
            et = metagraph.edge_types[code]
            rows = self.lookup(et.source, group["source"])
            cols = self.lookup(et.target, group["target"])
            self.matrices[code] = self.adjacency(code, rows, cols)

    def lookup(self, kind: str, ids: pd.Series) -> np.ndarray:
        idx = ids.map(self.index[kind])
        if idx.isna().any():
            sys.exit(f"{idx.isna().sum()} edge endpoints are not {kind} nodes in the node table.")
        return idx.to_numpy(dtype=np.int64)

    def adjacency(self, code: str, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
        et = self.metagraph.edge_types[code]
        shape = (len(self.ids[et.source]), len(self.ids[et.target]))
        m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
        if et.source == et.target and et.direction == "both":
            m = m + m.T  # undirected edges between nodes of the same kind
        return (m > 0).astype(np.float64)

    def with_edges(self, code: str, rows: np.ndarray, cols: np.ndarray) -> "Graph":
        """Shallow copy with one edge type replaced, e.g. the label edge by training pairs."""
        copy = object.__new__(Graph)
        copy.__dict__ = {**self.__dict__, "matrices": {**self.matrices, code: self.adjacency(code, rows, cols)}}
        return copy

    def hop_matrix(self, hop: Hop) -> sp.csr_matrix:
        if hop.code not in self.matrices:
            sys.exit(f"Edge type {hop.code} was not loaded.")
        m = self.matrices[hop.code]
        return m.T.tocsr() if hop.transpose else m


def degree_weight(m: sp.csr_matrix, damping: float) -> sp.csr_matrix:
    def inv_pow(deg):
        out = np.zeros_like(deg)
        out[deg > 0] = deg[deg > 0] ** -damping
        return out

    rows = inv_pow(np.asarray(m.sum(axis=1)).ravel())
    cols = inv_pow(np.asarray(m.sum(axis=0)).ravel())
    return (sp.diags(rows) @ m @ sp.diags(cols)).tocsr()


@dataclass
class Block:
    """One hop, or an X -> Y -> X hop pair whose walks back to the same X are dropped."""

    first: sp.csr_matrix
    second: sp.csr_matrix | None = None

    @property
    def n_cols(self) -> int:
        return (self.second if self.second is not None else self.first).shape[1]

    def apply(self, right: sp.csr_matrix) -> sp.csr_matrix:
        """self @ right, without forming the (possibly huge) X x X block."""
        if self.second is None:
            return self.first @ right
        back = np.asarray(self.first.multiply(self.second.T).sum(axis=1)).ravel()
        return self.first @ (self.second @ right) - sp.diags(back) @ right


def chain(blocks: list[Block]) -> sp.csr_matrix:
    # Right to left keeps intermediates narrow, since the last hop ends at Disease.
    out = sp.identity(blocks[-1].n_cols, format="csr")
    for b in reversed(blocks):
        out = b.apply(out)
    return out.tocsr()


def without_diagonal(m: sp.csr_matrix) -> sp.csr_matrix:
    m = (m - sp.diags(m.diagonal())).tocsr()
    m.eliminate_zeros()
    return m


class FeatureBuilder:
    def __init__(self, graph: Graph, ml_cfg: dict):
        self.graph = graph
        mg = graph.metagraph
        self.label_code = mg.abbrev(ml_cfg["label_edge"])
        label = mg.edge_types[self.label_code]
        self.drug_kind, self.disease_kind = label.source, label.target
        self.damping = ml_cfg["damping"]
        self.metapaths = {text: parse_metapath(text, mg) for text in ml_cfg["metapaths"]}
        for text, hops in self.metapaths.items():
            self._validate(text, hops)
        self.degree_specs = ml_cfg["degree_features"]
        for spec in self.degree_specs:
            et = mg.edge_types.get(spec["metaedge"])
            if et is None or spec["node"] not in (et.source, et.target):
                sys.exit(f"Bad degree feature {spec}: edge type unknown or node kind not on it.")
            if spec["node"] not in (self.drug_kind, self.disease_kind):
                sys.exit(f"Degree feature {spec} must be on {self.drug_kind} or {self.disease_kind}.")

    @staticmethod
    def required_codes(ml_cfg: dict, metagraph: Metagraph) -> set[str]:
        codes = {metagraph.abbrev(ml_cfg["label_edge"])}
        codes |= {metagraph.abbrev(s) for s in ml_cfg["exclude_from_negatives"]}
        codes |= {s["metaedge"] for s in ml_cfg["degree_features"]}
        for text in ml_cfg["metapaths"]:
            codes |= {h.code for h in parse_metapath(text, metagraph)}
        return codes

    def _validate(self, text: str, hops: list[Hop]) -> None:
        if hops[0].source != self.drug_kind or hops[-1].target != self.disease_kind:
            sys.exit(f"Metapath '{text}' must start at {self.drug_kind} and end at {self.disease_kind}.")
        label_hops = [i for i, h in enumerate(hops) if h.code == self.label_code]
        if len(hops) == 1 and label_hops:
            sys.exit(f"Metapath '{text}' is the label edge itself, which would leak the answer.")
        if len(label_hops) > 1 or any(i not in (0, len(hops) - 1) for i in label_hops):
            sys.exit(f"Metapath '{text}': the label edge may appear once, as the first or last hop.")

    def _blocks(self, hops: list[Hop], mats: list[sp.csr_matrix]) -> list[Block]:
        out, i = [], 0
        while i < len(hops):
            if i + 1 < len(hops) and hops[i].source == hops[i + 1].target:
                out.append(Block(mats[i], mats[i + 1]))
                i += 2
            else:
                out.append(Block(mats[i]))
                i += 1
        return out

    @property
    def feature_names(self) -> list[str]:
        names = [f"dwpc_{t}" for t in self.metapaths]
        names += [f"deg_{s['metaedge']}_{s['node']}" for s in self.degree_specs]
        return names

    def matrices(self, label_rows: np.ndarray, label_cols: np.ndarray,
                 scales: dict[str, float] | None = None) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """All features as (n_drugs x n_diseases) arrays, using only the given label pairs.

        Metapath values are transformed with asinh(x / scale). Pass `scales` from a
        reference graph so features computed on different label subsets (folds)
        share one scale. Returns (features, scales used).
        """
        g = self.graph.with_edges(self.label_code, label_rows, label_cols)
        out, used = {}, {}
        for text, hops in self.metapaths.items():
            mats = [degree_weight(g.hop_matrix(h), self.damping) for h in hops]
            label_hops = [i for i, h in enumerate(hops) if h.code == self.label_code]
            if not label_hops:
                m = chain(self._blocks(hops, mats))
            elif label_hops[0] == len(hops) - 1:
                m = without_diagonal(chain(self._blocks(hops[:-1], mats[:-1]))) @ mats[-1]
            else:
                m = mats[0] @ without_diagonal(chain(self._blocks(hops[1:], mats[1:])))
            arr = m.toarray()
            name = f"dwpc_{text}"
            scale = scales[name] if scales is not None else float(arr.mean())
            used[name] = scale
            out[name] = np.arcsinh(arr / scale) if scale > 0 else arr

        n_drug, n_dis = len(g.ids[self.drug_kind]), len(g.ids[self.disease_kind])
        for spec in self.degree_specs:
            m = g.matrices[spec["metaedge"]]
            et = g.metagraph.edge_types[spec["metaedge"]]
            axis = 1 if spec["node"] == et.source else 0
            deg = np.log1p(np.asarray(m.sum(axis=axis)).ravel())
            name = f"deg_{spec['metaedge']}_{spec['node']}"
            if spec["node"] == self.drug_kind:
                out[name] = np.repeat(deg[:, None], n_dis, axis=1)
            else:
                out[name] = np.repeat(deg[None, :], n_drug, axis=0)
        return out, used

    def frame(self, mats: dict[str, np.ndarray], rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({name: mats[name][rows, cols] for name in self.feature_names})
