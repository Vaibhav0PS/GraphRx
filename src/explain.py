"""Explain model predictions as concrete paths in Hetionet.

train.py lists, for each prediction, the metapaths that raised its score (the
"evidence" column, e.g. "CbGaD (+0.86)"). A metapath feature (DWPC) is a sum
over all walks of that pattern, where each walk weighs the product over its
hops of the degree-weighted edge weight. This script finds the walks behind
that number: for every prediction and every metapath in its evidence, the
`paths_per_metapath` concrete paths with the largest weight. Paths never
revisit a node.

Search: for the target disease we first compute, backwards along the
metapath, the best weight with which each node can still reach the disease
(0 = cannot reach). A forward beam search from the drug then keeps only nodes
that can reach the disease, ranked by their weight so far times that best
completion, `beam_width` partial paths per hop.

The graph is the one that scored the predictions: its label edges (e.g.
Compound-treats-Disease) are read from predictions.graph_label_edges_file,
written by train.py. That is all known label edges by default, or only the
development edges when ml.final.retrain_on_all_labels is false, so a path
never uses an edge the model did not see. Each output edge is in the
orientation stored in the edge file, with the relation name from the
metagraph.

Settings: ml.final.explain (paths_per_metapath, beam_width,
sanity_check_rows), ml.damping, predictions.file / evidence_column /
explanations_file in config.json.

Usage:
    python src/explain.py
    python src/explain.py --config other_config.json
"""

import argparse
import json
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

import hetionet
from config import load_config, raw_files, resolve
from features import FeatureBuilder, Graph, degree_weight

EVIDENCE_ITEM = re.compile(r"^(\S+)\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)$")


def parse_evidence(text: str) -> list[tuple[str, float]]:
    """'CbGaD (+0.86); CrCtD (+0.49)' -> [('CbGaD', 0.86), ('CrCtD', 0.49)]."""
    out = []
    for item in filter(None, (s.strip() for s in text.split(";"))):
        m = EVIDENCE_ITEM.match(item)
        if not m:
            sys.exit(f"Cannot parse evidence item '{item}'. Expected 'METAPATH (+value)'.")
        out.append((m.group(1), float(m.group(2))))
    return out


class PathFinder:
    """Strongest concrete paths that follow a metapath between two nodes."""

    def __init__(self, graph: Graph, builder: FeatureBuilder, edges: pd.DataFrame):
        self.graph = graph
        self.hops = builder.metapaths
        self.weights = {
            text: [degree_weight(graph.hop_matrix(h), builder.damping) for h in hops]
            for text, hops in self.hops.items()
        }
        self.kinds = {text: [h.source for h in hops] + [hops[-1].target] for text, hops in self.hops.items()}
        # Undirected edges between nodes of one kind (e.g. Gene-interacts-Gene) are
        # stored once, in either orientation; remember which one to report it as stored.
        self.stored: dict[str, set[tuple[str, str]]] = {}
        for code in {h.code for hops in self.hops.values() for h in hops}:
            et = graph.metagraph.edge_types[code]
            if et.source == et.target and et.direction == "both":
                sub = edges[edges["metaedge"] == code]
                self.stored[code] = set(zip(sub["source"], sub["target"]))
        self._target = None
        self._best: dict[str, list[np.ndarray]] = {}

    def best_completion(self, text: str, target: int) -> list[np.ndarray]:
        """best[i][u]: largest weight of a walk from node u at position i to the target (0 = unreachable)."""
        if target != self._target:  # callers go disease by disease, so one target is cached
            self._target, self._best = target, {}
        if text not in self._best:
            mats = self.weights[text]
            best = np.zeros(mats[-1].shape[1])
            best[target] = 1.0
            out = [best]
            for m in reversed(mats):
                best = (m @ sp.diags(best)).max(axis=1).toarray().ravel()
                out.append(best)
            self._best[text] = out[::-1]
        return self._best[text]

    def search(self, text: str, source: int, target: int, beam_width: int | None, k: int | None):
        """Top-k (node indexes, weight) paths, strongest first. None = no limit (exhaustive)."""
        mats, kinds = self.weights[text], self.kinds[text]
        best = self.best_completion(text, target)
        if best[0][source] <= 0:
            return []
        paths = np.array([[source]], dtype=np.int64)
        weight = np.ones(1)
        for i, m in enumerate(mats):
            starts = m.indptr[paths[:, -1]]
            counts = m.indptr[paths[:, -1] + 1] - starts
            parent = np.repeat(np.arange(len(paths)), counts)
            pos = np.arange(counts.sum()) + np.repeat(starts - (np.cumsum(counts) - counts), counts)
            nxt, val = m.indices[pos], m.data[pos]
            ok = best[i + 1][nxt] > 0
            for j in range(i + 1):
                if kinds[j] == kinds[i + 1]:  # no revisiting a node
                    ok &= nxt != paths[parent, j]
            parent, nxt = parent[ok], nxt[ok]
            new_weight = weight[parent] * val[ok]
            if beam_width is not None and len(nxt) > beam_width:
                keep = np.argsort(-(new_weight * best[i + 1][nxt]), kind="stable")[:beam_width]
                parent, nxt, new_weight = parent[keep], nxt[keep], new_weight[keep]
            if not len(nxt):
                return []
            paths = np.column_stack([paths[parent], nxt])
            weight = new_weight
        order = np.argsort(-weight, kind="stable")[:k]
        return [(paths[o], float(weight[o])) for o in order]

    def describe(self, text: str, path: np.ndarray) -> tuple[list[str], list[dict]]:
        """Node ids along the path, and its edges in stored orientation."""
        ids = [str(self.graph.ids[kind][i]) for kind, i in zip(self.kinds[text], path)]
        edges = []
        for hop, u, v in zip(self.hops[text], ids, ids[1:]):
            et = self.graph.metagraph.edge_types[hop.code]
            if hop.code in self.stored:
                s, t = (u, v) if (u, v) in self.stored[hop.code] else (v, u)
            else:
                s, t = (v, u) if hop.transpose else (u, v)
            edges.append({"source": s, "target": t, "rel": et.relation})
        return ids, edges


def scoring_graph(cfg: dict, graph: Graph, builder: FeatureBuilder, edges: pd.DataFrame) -> Graph:
    """The graph with its label edges replaced by those that scored the predictions."""
    path = resolve(cfg["predictions"]["graph_label_edges_file"])
    if not path.exists():
        sys.exit(f"No {path}.\nRun first:  python src/train.py")
    used = pd.read_csv(path, dtype=str)
    known = edges[edges["metaedge"] == builder.label_code]
    stray = set(zip(used["drug_id"], used["disease_id"])) - set(zip(known["source"], known["target"]))
    if stray:
        sys.exit(f"{path} has {len(stray)} pairs that are not {builder.label_code} edges. Rerun python src/train.py.")
    rows = graph.lookup(builder.drug_kind, used["drug_id"])
    cols = graph.lookup(builder.disease_kind, used["disease_id"])
    print(f"  Label edges in the scoring graph: {len(rows)} of {len(known)} known {builder.label_code} edges")
    return graph.with_edges(builder.label_code, rows, cols)


def raw_dwpc(builder: FeatureBuilder, graph: Graph) -> dict[str, np.ndarray]:
    """Untransformed DWPC per metapath on the full graph, for consistency checks."""
    rows, cols = graph.matrices[builder.label_code].nonzero()
    feats, scales = builder.matrices(rows, cols)
    out = {}
    for text in builder.metapaths:
        name = f"dwpc_{text}"
        out[text] = np.sinh(feats[name]) * scales[name] if scales[name] > 0 else feats[name]
    return out


def sig(x: float) -> float:
    return float(f"{x:.6g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Path to config file (default: config.json)")
    args = p.parse_args()

    cfg = load_config(args.config)
    ml, pcfg = cfg["ml"], cfg["predictions"]
    ecfg = ml["final"]["explain"]
    beam_width, per_metapath = ecfg["beam_width"], ecfg["paths_per_metapath"]
    if beam_width < 1 or per_metapath < 1:
        sys.exit("ml.final.explain.beam_width and paths_per_metapath must be at least 1.")
    t0 = time.time()

    pred_path = resolve(pcfg["file"])
    if not pred_path.exists():
        sys.exit(f"No predictions at {pred_path}.\nRun first:  python src/train.py")
    columns = [pcfg["drug_column"], pcfg["disease_column"], pcfg["evidence_column"]]
    preds = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
    missing = set(columns) - set(preds.columns)
    if missing:
        sys.exit(f"{pred_path} is missing columns: {', '.join(sorted(missing))}")

    metagraph = hetionet.Metagraph(raw_files(cfg)["metagraph"])
    nodes, edges, metagraph = hetionet.load(cfg, codes=FeatureBuilder.required_codes(ml, metagraph))
    graph = Graph(nodes, edges, metagraph)
    builder = FeatureBuilder(graph, ml)
    graph = scoring_graph(cfg, graph, builder, edges)
    builder.graph = graph
    finder = PathFinder(graph, builder, edges)
    dwpc = raw_dwpc(builder, graph)

    drug_index, disease_index = graph.index[builder.drug_kind], graph.index[builder.disease_kind]
    unknown_ids = (set(preds[columns[0]]) - set(drug_index)) | (set(preds[columns[1]]) - set(disease_index))
    if unknown_ids:
        sys.exit(f"{len(unknown_ids)} ids in {pred_path} are not {builder.drug_kind}/{builder.disease_kind} "
                 f"nodes, e.g. {sorted(unknown_ids)[0]}. Rerun python src/train.py.")
    print(f"Explaining {len(preds)} predictions (beam width {beam_width}, "
          f"{per_metapath} paths per metapath) ...")

    results: dict[int, dict] = {}
    checked, inconsistent, skipped = 0, [], set()
    order = preds.assign(_c=preds[columns[1]].map(disease_index)).sort_values("_c", kind="stable").index
    for i in order:
        row = preds.loc[i]
        r, c = drug_index[row[columns[0]]], disease_index[row[columns[1]]]
        paths = []
        for text, contribution in parse_evidence(row[columns[2]]):
            if text not in builder.metapaths:
                skipped.add(text)
                continue
            found = finder.search(text, r, c, beam_width, per_metapath)
            total = float(dwpc[text][r, c])
            checked += 1
            top_sum = sum(w for _, w in found)
            if (total > 0) != bool(found) or top_sum > total * (1 + 1e-6):
                inconsistent.append((row[columns[0]], row[columns[1]], text, total, top_sum))
            for path, w in found:
                ids, path_edges = finder.describe(text, path)
                paths.append({
                    "metapath": text,
                    "contribution": contribution,
                    "weight": sig(w),
                    "share": sig(w / total) if total > 0 else None,
                    "nodes": ids,
                    "edges": path_edges,
                })
        results[i] = {"drug_id": row[columns[0]], "disease_id": row[columns[1]], "paths": paths}

    out = [results[i] for i in preds.index]
    out_path = resolve(pcfg["explanations_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    no_paths = [x for x in out if not x["paths"]]
    no_evidence = sum(1 for i in preds.index if not parse_evidence(preds.at[i, columns[2]]))
    by_metapath = pd.Series([p["metapath"] for x in out for p in x["paths"]]).value_counts()
    print(f"  Rows explained: {len(out) - len(no_paths)} of {len(out)}; "
          f"rows with no paths: {len(no_paths)} ({no_evidence} have no evidence metapaths)")
    print(f"  Paths: {int(by_metapath.sum())}  " + ", ".join(f"{k}={v}" for k, v in by_metapath.items()))
    if skipped:
        print(f"  Warning: evidence metapaths not in config.json were skipped: {', '.join(sorted(skipped))}")

    # --- consistency with the model's features
    print(f"  Check: {checked} (prediction, metapath) pairs, {len(inconsistent)} where paths found "
          f"disagree with DWPC > 0 or exceed it")
    for drug, disease, text, total, top_sum in inconsistent[:5]:
        print(f"    {drug} -> {disease} {text}: DWPC={total:.4g}, top paths sum={top_sum:.4g}")
    n_check = min(ecfg["sanity_check_rows"], len(preds))
    if n_check:
        print(f"  Exhaustive check on {n_check} predictions (all paths, no beam): DWPC vs sum over paths")
        rng = np.random.default_rng(ml["seed"])
        name_of = dict(zip(nodes["id"], nodes["name"]))
        for i in sorted(rng.choice(preds.index.to_numpy(), size=n_check, replace=False)):
            row = preds.loc[i]
            r, c = drug_index[row[columns[0]]], disease_index[row[columns[1]]]
            for text, _ in parse_evidence(row[columns[2]]):
                if text not in builder.metapaths:
                    continue
                every = finder.search(text, r, c, None, None)
                top = finder.search(text, r, c, beam_width, per_metapath)
                total = float(dwpc[text][r, c])
                exact = sum(w for _, w in every)
                beam_ok = [round(w, 12) for _, w in top] == [round(w, 12) for _, w in every[:per_metapath]]
                print(f"    {name_of[row[columns[0]]]} -> {name_of[row[columns[1]]]} {text}: "
                      f"DWPC={total:.4g}, {len(every)} paths sum={exact:.4g} "
                      f"({exact / total:.3f} of DWPC), top-{per_metapath} share="
                      f"{sum(w for _, w in top) / total:.3f}, beam = exhaustive top: {beam_ok}")

    print(f"Wrote {out_path} in {time.time() - t0:.1f}s")
    print("Next: python src/build_subgraph.py")


if __name__ == "__main__":
    main()
