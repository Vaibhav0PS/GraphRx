"""Build a small, viewable subgraph of Hetionet around seed diseases.

The full graph is too big to draw usefully, so we keep the parts that matter
for drug repurposing. Edge types come from the "schema" section of
config.json:

  label_edges       drug -> disease   known treatments (the ML labels)
  target_edge       drug -> target    which targets a drug binds
  association_edge  disease - target  targets linked to a disease
  context_edge      target - context  e.g. pathways the targets belong to

Only targets that link an included drug to a seed disease are kept, and only
context nodes shared by several of those targets.

"Repurposing candidates" are drugs with no label edge to a disease that bind
at least `min_shared_targets` of its associated targets. This is a simple
heuristic, not an ML prediction. Model predictions can be overlaid with
--predictions.

If the explanations file from src/explain.py exists (and
predictions.show_explanation_paths is true), the paths that explain each
shown prediction are added to the subgraph and attached to its link, so the
viewer can show why it was predicted.

Usage:
    python src/build_subgraph.py
    python src/build_subgraph.py --diseases "asthma" "psoriasis"
    python src/build_subgraph.py --list-diseases
    python src/build_subgraph.py --predictions results/predictions.csv
"""

import argparse
import difflib
import json
import sys

import pandas as pd

import hetionet
from config import load_config, resolve
from features import parse_metapath

ROLE_SEED = "seed"
ROLE_KNOWN = "known"
ROLE_CANDIDATE = "candidate"
ROLE_PREDICTED = "predicted"


def oriented(edges: pd.DataFrame, metagraph: hetionet.Metagraph, spec: list[str], first_kind: str) -> pd.DataFrame:
    """Edges of one type as columns (a, b) with `a` always the node of `first_kind`."""
    rows = edges[edges["metaedge"] == metagraph.abbrev(spec)]
    if spec[0] == first_kind:
        return rows.assign(a=rows["source"], b=rows["target"])
    if spec[2] == first_kind:
        return rows.assign(a=rows["target"], b=rows["source"])
    sys.exit(f"Edge type {spec} has no end of kind '{first_kind}'. Check the schema in config.json.")


def schema_kinds(schema: dict) -> dict[str, str]:
    """Work out which node kind plays each role, and check the schema is consistent."""
    drug, _, disease = schema["label_edges"][0]
    for spec in schema["label_edges"]:
        if (spec[0], spec[2]) != (drug, disease):
            sys.exit(f"All label_edges must be {drug} -> {disease}, got {spec}")
    if schema["target_edge"][0] != drug:
        sys.exit(f"target_edge must start at {drug}, got {schema['target_edge']}")
    target = schema["target_edge"][2]
    assoc = schema["association_edge"]
    if {assoc[0], assoc[2]} != {disease, target}:
        sys.exit(f"association_edge must connect {disease} and {target}, got {assoc}")
    ctx = schema["context_edge"]
    if target not in (ctx[0], ctx[2]):
        sys.exit(f"context_edge must touch {target}, got {ctx}")
    context = ctx[2] if ctx[0] == target else ctx[0]
    return {"drug": drug, "disease": disease, "target": target, "context": context}


def pick_seeds(nodes: pd.DataFrame, labels: pd.DataFrame, kinds: dict, names: list[str], auto_count: int) -> list[str]:
    diseases = nodes[nodes["kind"] == kinds["disease"]]
    if not names:
        # No seeds given: take the diseases with the most known treatments.
        counts = labels.groupby("b").size().rename("n").reset_index()
        counts = counts.merge(diseases, left_on="b", right_on="id")
        counts = counts.sort_values(["n", "name"], ascending=[False, True]).head(auto_count)
        print(f"  Auto-selected {len(counts)} seed diseases with the most known treatments")
        return list(counts["id"])

    by_name = {n.lower(): i for n, i in zip(diseases["name"], diseases["id"])}
    ids = []
    for name in names:
        if name.lower() in by_name:
            ids.append(by_name[name.lower()])
            continue
        close = difflib.get_close_matches(name.lower(), by_name.keys(), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        print(f"  Warning: '{name}' not found.{hint}")
    if not ids:
        sys.exit("No valid seed diseases. Use --list-diseases to see the options.")
    return ids


def load_predictions(cfg: dict, path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    cols = {cfg["drug_column"]: "a", cfg["disease_column"]: "b", cfg["score_column"]: "score"}
    missing = set(cols) - set(df.columns)
    if missing:
        sys.exit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    df = df.rename(columns=cols)[["a", "b", "score"]]
    df["score"] = df["score"].astype(float)
    return df.sort_values("score", ascending=False).groupby("b").head(cfg["top_per_disease"])


def load_explanations(path) -> dict[tuple[str, str], list[dict]]:
    """(drug id, disease id) -> explanation paths, as written by src/explain.py."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {(r["drug_id"], r["disease_id"]): r["paths"] for r in rows}


def is_symmetric(et: hetionet.EdgeType) -> bool:
    """Undirected edge between two nodes of one kind, e.g. Gene-interacts-Gene."""
    return et.source == et.target and et.direction == "both"


def describe_metapath(text: str, metagraph: hetionet.Metagraph) -> str:
    """'CbGaD' -> 'Compound —binds→ Gene ←associates— Disease' (arrows follow the stored edge)."""
    hops = parse_metapath(text, metagraph)
    parts = [hops[0].source]
    for h in hops:
        et = metagraph.edge_types[h.code]
        if is_symmetric(et):
            parts.append(f"—{et.relation}—")
        elif h.transpose:
            parts.append(f"←{et.relation}—")
        else:
            parts.append(f"—{et.relation}→")
        parts.append(h.target)
    return " ".join(parts)


def build(nodes, edges, metagraph, cfg, seed_names, predictions, explanations=None) -> dict:
    schema, sub = cfg["schema"], cfg["subgraph"]
    kinds = schema_kinds(schema)

    labels = pd.concat(
        [oriented(edges, metagraph, s, kinds["drug"]).assign(rel=s[1]) for s in schema["label_edges"]],
        ignore_index=True,
    )
    seeds = pick_seeds(nodes, labels, kinds, seed_names, sub["auto_seed_count"])
    seed_set = set(seeds)

    labels = labels[labels["b"].isin(seed_set)]
    known_by_disease = labels.groupby("b")["a"].apply(set).to_dict()

    assoc = oriented(edges, metagraph, schema["association_edge"], kinds["disease"])
    assoc = assoc[assoc["a"].isin(seed_set)]
    targets_by_disease = assoc.groupby("a")["b"].apply(set).to_dict()

    binds = oriented(edges, metagraph, schema["target_edge"], kinds["drug"])
    binds = binds[binds["b"].isin(set(assoc["b"]))]

    candidate_for: dict[str, list[dict]] = {}
    for d in seeds:
        hits = binds[binds["b"].isin(targets_by_disease.get(d, set()))]
        counts = hits.groupby("a").size()
        counts = counts.drop(labels=list(known_by_disease.get(d, set())), errors="ignore")
        counts = counts[counts >= sub["min_shared_targets"]]
        counts = counts.sort_values(ascending=False, kind="stable").head(sub["top_candidates"])
        for drug, n in counts.items():
            candidate_for.setdefault(drug, []).append({"disease_id": d, "shared_targets": int(n)})

    known = set(labels["a"])
    drugs = known | set(candidate_for)
    binds = binds[binds["a"].isin(drugs)]
    targets = set(binds["b"])
    assoc = assoc[assoc["b"].isin(targets)]

    ctx = oriented(edges, metagraph, schema["context_edge"], kinds["target"])
    ctx = ctx[ctx["a"].isin(targets)]
    ctx_counts = ctx.groupby("b").size()
    ctx_counts = ctx_counts[ctx_counts >= sub["min_context_targets"]]
    context = set(ctx_counts.sort_values(ascending=False, kind="stable").head(sub["max_context_nodes"]).index)
    ctx = ctx[ctx["b"].isin(context)]

    frames = [
        labels[["source", "target", "rel"]],
        binds[["source", "target"]].assign(rel=schema["target_edge"][1]),
        assoc[["source", "target"]].assign(rel=schema["association_edge"][1]),
        ctx[["source", "target"]].assign(rel=schema["context_edge"][1]),
    ]
    predicted, path_nodes = set(), set()
    explained: dict[tuple[str, str], list[dict]] = {}
    pred_rel = cfg["predictions"]["relation"]
    if predictions is not None:
        pred = predictions[predictions["b"].isin(seed_set)]
        predicted = set(pred["a"])
        if explanations is not None:
            path_edges = []
            for pair in zip(pred["a"], pred["b"]):
                paths = explanations.get(pair, [])
                explained[pair] = [
                    {k: p[k] for k in ("metapath", "contribution", "share", "nodes", "edges") if k in p} for p in paths
                ]
                for p in paths:
                    path_nodes.update(p["nodes"])
                    path_edges += p["edges"]
            if path_edges:
                # Edges are in stored orientation, so duplicates of edges already kept drop out below.
                frames.append(pd.DataFrame(path_edges, columns=["source", "target", "rel"]))
        frames.append(pred.rename(columns={"a": "source", "b": "target"}).assign(rel=pred_rel))

    links = pd.concat(frames, ignore_index=True).drop_duplicates(["source", "target", "rel"])

    keep = seed_set | drugs | targets | context | predicted | path_nodes
    node_df = nodes[nodes["id"].isin(keep)]
    missing = keep - set(node_df["id"])
    if missing:
        print(f"  Warning: {len(missing)} ids not in the node table, dropping their edges")
        links = links[~links["source"].isin(missing) & ~links["target"].isin(missing)]

    name_of = dict(zip(nodes["id"], nodes["name"]))
    out_nodes = []
    for row in node_df.itertuples(index=False):
        node = {"id": row.id, "name": row.name, "kind": row.kind}
        if row.id in seed_set:
            node["role"] = ROLE_SEED
        elif row.id in known:
            node["role"] = ROLE_KNOWN
        elif row.id in candidate_for:
            node["role"] = ROLE_CANDIDATE
        elif row.id in predicted:
            node["role"] = ROLE_PREDICTED
        if row.id in candidate_for:
            node["candidate_for"] = [{**c, "disease": name_of[c["disease_id"]]} for c in candidate_for[row.id]]
        out_nodes.append(node)

    out_links = []
    for row in links.itertuples(index=False):
        link = {"source": row.source, "target": row.target, "rel": row.rel}
        if row.rel == pred_rel:
            link["score"] = round(float(row.score), 4)
            if explanations is not None:
                link["explanations"] = explained.get((row.source, row.target), [])
        out_links.append(link)

    meta_extra = {}
    if explanations is not None:
        used = sorted({p["metapath"] for paths in explained.values() for p in paths})
        kinds_used = set(node_df["kind"])
        meta_extra = {
            "metapaths": {t: describe_metapath(t, metagraph) for t in used},
            # [kind, relation, kind] edge types whose direction carries no meaning
            "symmetric_edges": sorted(
                [et.source, et.relation, et.target]
                for et in metagraph.edge_types.values() if is_symmetric(et) and et.source in kinds_used
            ),
            "explained_links": sum(1 for paths in explained.values() if paths),
            "explanation_paths": sum(len(paths) for paths in explained.values()),
        }

    return {
        "meta": {
            "source": cfg["data"]["base_url"],
            "seed_diseases": [name_of[d] for d in seeds],
            "kinds": kinds,
            "label_relations": [s[1] for s in schema["label_edges"]],
            "prediction_relation": pred_rel,
            "roles": {"seed": ROLE_SEED, "known": ROLE_KNOWN, "candidate": ROLE_CANDIDATE, "predicted": ROLE_PREDICTED},
            "node_count": len(out_nodes),
            "link_count": len(out_links),
            **meta_extra,
        },
        "nodes": out_nodes,
        "links": out_links,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Path to config file (default: config.json)")
    p.add_argument("--diseases", nargs="+", help="Seed disease names (overrides config)")
    p.add_argument("--list-diseases", action="store_true", help="Print all disease names and exit")
    p.add_argument("--predictions", help="CSV of model scores to overlay (overrides config)")
    args = p.parse_args()

    cfg = load_config(args.config)
    schema = cfg["schema"]
    specs = schema["label_edges"] + [schema["target_edge"], schema["association_edge"], schema["context_edge"]]
    nodes, edges, metagraph = hetionet.load(cfg, specs)

    if args.list_diseases:
        disease_kind = schema_kinds(schema)["disease"]
        for name in sorted(nodes.loc[nodes["kind"] == disease_kind, "name"], key=str.lower):
            print(name)
        return

    seed_names = args.diseases if args.diseases else cfg["subgraph"]["seed_diseases"]
    pred_file = args.predictions or cfg["predictions"]["file"]
    predictions = None
    if pred_file and resolve(pred_file).exists():
        predictions = load_predictions(cfg["predictions"], resolve(pred_file))
        print(f"  Overlaying predictions from {resolve(pred_file)}")
    elif pred_file:
        print(f"  No predictions at {resolve(pred_file)} yet (run src/train.py); building without them")

    explanations = None
    pcfg = cfg["predictions"]
    if predictions is not None and pcfg["show_explanation_paths"]:
        exp_file = resolve(pcfg["explanations_file"])
        if exp_file.exists():
            explanations = load_explanations(exp_file)
            print(f"  Adding explanation paths from {exp_file}")
        else:
            print(f"  No explanations at {exp_file} (run python src/explain.py); building without them")

    graph = build(nodes, edges, metagraph, cfg, seed_names, predictions, explanations)

    out_dir = resolve(cfg["subgraph"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(graph, ensure_ascii=False)
    (out_dir / "graph.json").write_text(payload, encoding="utf-8")
    # A .js copy lets index.html open straight from disk without a web server.
    (out_dir / "graph.js").write_text(f"window.GRAPH_DATA = {payload};\n", encoding="utf-8")

    by_kind = pd.Series([n["kind"] for n in graph["nodes"]]).value_counts().to_dict()
    print(f"Seeds: {', '.join(graph['meta']['seed_diseases'])}")
    print(f"Subgraph: {graph['meta']['node_count']} nodes, {graph['meta']['link_count']} links {by_kind}")
    if explanations is not None:
        n_pred = sum(1 for link in graph["links"] if link["rel"] == graph["meta"]["prediction_relation"])
        print(f"Explanations: {graph['meta']['explanation_paths']} paths on "
              f"{graph['meta']['explained_links']} of {n_pred} predicted links")
    print(f"Wrote {out_dir / 'graph.js'}")


if __name__ == "__main__":
    main()
