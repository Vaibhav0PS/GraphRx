"""R-GCN (relational graph convolutional network) for drug-disease link prediction.

Run AFTER `python src/train.py`. This script reads split.csv, dev_negatives.csv,
metrics.json and test_scores.csv.gz from ml.results_dir and adds itself as one
more model to metrics.json and test_scores.csv.gz. Re-running train.py
overwrites those files, so re-run this script after it (then src/plots.py).
Re-running this script replaces only its own entry and column.

Model (settings in the "gnn" section of config.json):
  - A learnable embedding per node, then `num_layers` R-GCN layers over the
    Hetionet edge types in `graph_edges` minus `graph_exclude`. Each edge type
    gives one relation per direction (one relation for undirected edges between
    nodes of the same kind). Relation weights use a basis decomposition with
    `num_bases` bases. Each layer has a self-loop, mean aggregation per relation,
    and ReLU + dropout between layers.
  - A DistMult (or full bilinear) decoder scores every drug x disease pair.
  - Loss: binary cross-entropy on supervision positives and sampled negatives.

Leakage control:
  - Test positives (split "test" in split.csv) are never in the graph or the
    supervision. The label edge type in the graph is rebuilt from allowed
    training pairs only; the full label matrix is dropped after loading.
  - Model selection uses ONE held-out fold (`validation_fold`, e.g. "cv0"), not
    5-fold CV. Its positives are removed from the graph and the supervision and
    ranked against `val_negative_fraction` of the unknown pairs outside
    dev_negatives.csv, which are never trained on. Use 1.0 to match train.py's
    CV scale; AUPRC rises as negatives are subsampled. The grid point and epoch with
    the best ml.selection_metric are kept (early stopping with `patience`
    evaluations, one every `eval_every` epochs).
  - Each epoch the training positives are split at random into message-passing
    edges (in the graph) and supervision edges (`supervision_fraction`, scored by
    the decoder), so no positive is ever predicted through its own edge.
  - Training negatives are sampled only from dev_negatives.csv, the negatives
    train.py used for its test models. The test negatives are every other unknown
    pair, exactly as `eval_neg` in train.py.
  - The test model is retrained on all development positives for the selected
    number of epochs and scored with all development label edges in the graph.

Outputs in ml.results_dir:
    metrics.json        models[<model_name>] (validation, test, CI, subsets, training info)
    test_scores.csv.gz  column <model_name> (raw decoder logits; higher = more likely)
    gnn_history.csv     loss and validation metric per epoch for every run

Usage:
    python src/train.py          (first)
    python src/train_gnn.py [--config CONFIG]
    python src/plots.py
"""

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

import hetionet
from config import load_config, raw_files, resolve
from features import Graph
from train import bootstrap_ci, evaluate, fmt, grid, pair_mask

REQUIRED_RESULTS = ["split.csv", "dev_negatives.csv", "metrics.json", "test_scores.csv.gz"]
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")


# ---------------------------------------------------------------- sparse relations

class SpMM(torch.autograd.Function):
    """A @ X for a fixed sparse A. The backward pass uses a precomputed A^T."""

    @staticmethod
    def forward(ctx, a, a_t, x):
        ctx.a_t = a_t
        return a @ x

    @staticmethod
    def backward(ctx, grad):
        return None, None, ctx.a_t @ grad


def to_torch(m: sp.spmatrix) -> torch.Tensor:
    m = m.tocsr()
    m.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(m.indptr.astype(np.int64)),
        torch.from_numpy(m.indices.astype(np.int64)),
        torch.from_numpy(m.data.astype(np.float32)),
        size=m.shape,
    )


@dataclass(eq=False)
class Relation:
    """Messages from nodes of kind `src` to nodes of kind `tgt`, mean-normalized."""

    name: str
    src: int
    tgt: int
    a: torch.Tensor | None = None    # (n_tgt x n_src)
    a_t: torch.Tensor | None = None
    n_edges: int = 0

    def set(self, m: sp.spmatrix) -> None:
        """m: 0/1 matrix with rows = receivers, columns = senders."""
        m = sp.csr_matrix(m)
        deg = np.asarray(m.sum(axis=1)).ravel()
        inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
        norm = (sp.diags(inv) @ m).tocsr()
        self.a, self.a_t, self.n_edges = to_torch(norm), to_torch(norm.T), int(m.nnz)


def relations_for(code: str, metagraph: hetionet.Metagraph, kind_index: dict[str, int]) -> list[Relation]:
    et = metagraph.edge_types[code]
    s, t = kind_index[et.source], kind_index[et.target]
    if et.source == et.target and et.direction == "both":
        return [Relation(code, s, t)]  # Graph stores these symmetrically: one relation
    return [Relation(code, s, t), Relation(f"{code}^-1", t, s)]


def set_edges(rels: list[Relation], m: sp.spmatrix) -> None:
    """m as stored in features.Graph: rows = edge source kind, columns = target kind."""
    rels[0].set(m.T)
    if len(rels) == 2:
        rels[1].set(m)


# ---------------------------------------------------------------- model

class RGCNLayer(nn.Module):
    def __init__(self, dim: int, n_rel: int, n_bases: int):
        super().__init__()
        self.bases = nn.Parameter(torch.empty(n_bases, dim, dim))
        self.coef = nn.Parameter(torch.empty(n_rel, n_bases))
        self.loop = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coef)

    def forward(self, h: list, relations: list[Relation], need: set[int]) -> list:
        w = torch.einsum("rb,bio->rio", self.coef, self.bases)
        out = [self.loop(x) if k in need else None for k, x in enumerate(h)]
        for i, r in enumerate(relations):
            if r.tgt not in need or r.n_edges == 0:
                continue
            if r.a.shape[0] <= r.a.shape[1]:   # aggregate first when receivers are fewer
                msg = SpMM.apply(r.a, r.a_t, h[r.src]) @ w[i]
            else:
                msg = SpMM.apply(r.a, r.a_t, h[r.src] @ w[i])
            out[r.tgt] = out[r.tgt] + msg
        return out


class RGCN(nn.Module):
    def __init__(self, sizes: list[int], relations: list[Relation], out_kinds: tuple[int, int],
                 hidden_dim: int, num_layers: int, dropout: float, num_bases: int, decoder: str):
        super().__init__()
        self.relations = relations
        self.out_kinds = out_kinds
        self.dropout = dropout
        self.emb = nn.ParameterList([nn.Parameter(torch.randn(n, hidden_dim) / math.sqrt(hidden_dim)) for n in sizes])
        self.layers = nn.ModuleList([RGCNLayer(hidden_dim, len(relations), num_bases) for _ in range(num_layers)])
        # Node kinds each layer must output, working back from the decoder's two kinds.
        need = [set(out_kinds)]
        for _ in range(num_layers - 1):
            need.insert(0, need[0] | {r.src for r in relations if r.tgt in need[0] and r.n_edges})
        self.need = need
        if decoder == "distmult":
            self.rel = nn.Parameter(torch.ones(hidden_dim))
        elif decoder == "bilinear":
            self.rel = nn.Parameter(torch.eye(hidden_dim))
        else:
            sys.exit(f"Unknown gnn.decoder '{decoder}'. Use distmult or bilinear.")
        self.decoder = decoder

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        h = list(self.emb)
        for i, layer in enumerate(self.layers):
            h = layer(h, self.relations, self.need[i])
            if i < len(self.layers) - 1:
                h = [None if x is None else F.dropout(F.relu(x), self.dropout, self.training) for x in h]
        return h[self.out_kinds[0]], h[self.out_kinds[1]]

    def score(self, drugs: torch.Tensor, diseases: torch.Tensor, rows=None, cols=None) -> torch.Tensor:
        left = drugs * self.rel if self.decoder == "distmult" else drugs @ self.rel
        if rows is None:
            return left @ diseases.T
        return (left[rows] * diseases[cols]).sum(dim=1)


# ---------------------------------------------------------------- training

@dataclass
class Context:
    sizes: list[int]
    relations: list[Relation]
    label_rels: list[Relation]      # the label edge's relations (empty if not in the graph)
    label_shape: tuple[int, int]
    out_kinds: tuple[int, int]
    kinds: tuple[str, str]
    gnn: dict
    ml: dict

    def set_label_edges(self, rows: np.ndarray, cols: np.ndarray) -> None:
        if self.label_rels:
            m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=self.label_shape)
            set_edges(self.label_rels, (m > 0).astype(np.float64))


def build_model(ctx: Context, params: dict) -> RGCN:
    return RGCN(ctx.sizes, ctx.relations, ctx.out_kinds, params["hidden_dim"], params["num_layers"],
                params["dropout"], ctx.gnn["num_bases"], ctx.gnn["decoder"])


def predict(model: RGCN, ctx: Context, graph_pos: tuple) -> np.ndarray:
    """Logits for every drug x disease pair, with `graph_pos` as the label edges in the graph."""
    ctx.set_label_edges(*graph_pos)
    model.eval()
    with torch.no_grad():
        return model.score(*model()).double().numpy()


def fit(ctx: Context, params: dict, train_pos: tuple, neg_pool: tuple, epochs: int, run: str,
        val: tuple | None = None):
    """Train for up to `epochs`. With `val` = (positives, negative mask), evaluate every
    eval_every epochs and stop after `patience` evaluations without improvement.
    Returns (model, history rows, best validation result or None)."""
    gnn, ml = ctx.gnn, ctx.ml
    sel = ml["selection_metric"]
    torch.manual_seed(gnn["seed"])
    rng = np.random.default_rng(gnn["seed"])
    model = build_model(ctx, params)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    pr, pc = train_pos
    n_sup = max(1, round(gnn["supervision_fraction"] * len(pr)))
    n_neg = gnn["negatives_per_positive"] * n_sup
    if n_neg > len(neg_pool[0]):
        print(f"  note: {n_neg} negatives per epoch requested, pool has {len(neg_pool[0])}; using all")
        n_neg = len(neg_pool[0])
    y = torch.cat([torch.ones(n_sup), torch.zeros(n_neg)])

    history, best, bad = [], None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(pr))
        sup, mp = perm[:n_sup], perm[n_sup:]
        ctx.set_label_edges(pr[mp], pc[mp])     # supervision edges are NOT in the graph
        neg = rng.choice(len(neg_pool[0]), size=n_neg, replace=False)
        rows = torch.from_numpy(np.r_[pr[sup], neg_pool[0][neg]])
        cols = torch.from_numpy(np.r_[pc[sup], neg_pool[1][neg]])
        loss = F.binary_cross_entropy_with_logits(model.score(*model(), rows, cols), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        row = {"run": run, "params": json.dumps(params), "epoch": epoch, "loss": loss.item(),
               f"val_{sel}": np.nan, "elapsed_s": round(time.time() - t0, 2)}
        history.append(row)
        last = epoch == epochs
        if epoch % gnn["eval_every"] and not last:
            continue
        msg = f"    {run} epoch {epoch:4d}  loss {loss.item():.4f}  {time.time() - t0:6.1f}s"
        if val is None:
            print(msg)
            continue
        m = evaluate(predict(model, ctx, train_pos), val[0], val[1], ml["ranking_ks"], ctx.kinds)
        row[f"val_{sel}"] = m[sel]
        print(f"{msg}  val {sel} {m[sel]:.4f}")
        if best is None or m[sel] > best["metrics"][sel]:
            best, bad = {"epoch": epoch, "metrics": m}, 0
        else:
            bad += 1
            if bad >= gnn["patience"]:
                print(f"    early stop: no improvement in {bad} evaluations, best epoch {best['epoch']}")
                break
    return model, history, best


# ---------------------------------------------------------------- data

def graph_codes(gnn: dict, metagraph: hetionet.Metagraph, present: set[str]) -> list[str]:
    spec = gnn.get("graph_edges", "all")
    codes = sorted(present) if spec == "all" else [metagraph.abbrev(s) for s in spec]
    drop = {metagraph.abbrev(s) for s in gnn.get("graph_exclude", [])}
    return [c for c in codes if c not in drop]


def mask_from(graph: Graph, frame: pd.DataFrame, kinds: tuple, shape: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = graph.lookup(kinds[0], frame["drug_id"])
    c = graph.lookup(kinds[1], frame["disease_id"])
    m = np.zeros(shape, dtype=bool)
    m[r, c] = True
    return m, r, c


def prepare(cfg: dict, rng) -> SimpleNamespace:
    """Load the graph, re-create train.py's split and negatives, check them against
    test_scores.csv.gz, and build the message-passing relations."""
    ml, gnn = cfg["ml"], cfg["gnn"]
    out_dir = resolve(ml["results_dir"])
    missing = [f for f in REQUIRED_RESULTS if not (out_dir / f).exists()]
    if missing:
        sys.exit(f"Missing {missing} in {out_dir}. Run `python src/train.py` first.")

    # --- graph and labels (same definitions as train.py)
    metagraph = hetionet.Metagraph(raw_files(cfg)["metagraph"])
    nodes, edges, metagraph = hetionet.load(cfg)
    graph = Graph(nodes, edges, metagraph)
    label_code = metagraph.abbrev(ml["label_edge"])
    label_et = metagraph.edge_types[label_code]
    kinds = (label_et.source, label_et.target)
    shape = (len(graph.ids[kinds[0]]), len(graph.ids[kinds[1]]))
    positive = pair_mask(graph, edges, label_code, kinds, shape)
    excluded = np.zeros(shape, dtype=bool)
    for spec in ml["exclude_from_negatives"]:
        excluded |= pair_mask(graph, edges, metagraph.abbrev(spec), kinds, shape)
    excluded &= ~positive
    unknown = ~(positive | excluded)
    graph.matrices.pop(label_code)  # never used again: the label edge is rebuilt from allowed pairs

    # --- split written by train.py
    split = pd.read_csv(out_dir / "split.csv")
    split_mask, sr, sc = mask_from(graph, split, kinds, shape)
    if len(split) != positive.sum() or split.duplicated(["drug_id", "disease_id"]).any() or not (split_mask == positive).all():
        sys.exit("split.csv does not match the label edges in the data. Re-run `python src/train.py`.")
    is_test = (split["split"] == "test").to_numpy()
    is_val = (split["split"] == gnn["validation_fold"]).to_numpy()
    if not is_val.any():
        sys.exit(f"gnn.validation_fold '{gnn['validation_fold']}' is not a split in split.csv.")
    test_pos = (sr[is_test], sc[is_test])
    dev_pos = (sr[~is_test], sc[~is_test])
    fit_pos = (sr[~is_test & ~is_val], sc[~is_test & ~is_val])
    val_pos = (sr[is_val], sc[is_val])

    dev_neg, _, _ = mask_from(graph, pd.read_csv(out_dir / "dev_negatives.csv"), kinds, shape)
    if not unknown[dev_neg].all():
        sys.exit("dev_negatives.csv contains known pairs. Re-run `python src/train.py`.")
    eval_neg = unknown & ~dev_neg          # test negatives, exactly as in train.py
    neg_pool = np.nonzero(dev_neg)         # the only pairs ever used as training negatives

    # --- test_scores.csv.gz must hold exactly the test positives + eval_neg
    frame = pd.read_csv(out_dir / "test_scores.csv.gz")
    in_frame, fr, fc = mask_from(graph, frame, kinds, shape)
    expected = eval_neg.copy()
    expected[test_pos] = True
    test_mask = np.zeros(shape, dtype=bool)
    test_mask[test_pos] = True
    if (len(frame) != expected.sum() or not (in_frame == expected).all()
            or not ((frame["label"].to_numpy() == 1) == test_mask[fr, fc]).all()):
        sys.exit("test_scores.csv.gz does not match split.csv / dev_negatives.csv. Re-run `python src/train.py`.")

    # --- validation negatives: a fraction of the unknown pairs never trained on
    cand = np.flatnonzero(eval_neg)
    pick = rng.choice(cand, size=round(gnn["val_negative_fraction"] * len(cand)), replace=False)
    val_neg = np.zeros(shape, dtype=bool)
    val_neg.flat[pick] = True

    # --- relations
    kind_names = sorted(graph.ids)
    kind_index = {k: i for i, k in enumerate(kind_names)}
    codes = graph_codes(gnn, metagraph, set(edges["metaedge"]))
    relations, label_rels = [], []
    for code in codes:
        rels = relations_for(code, metagraph, kind_index)
        if code == label_code:
            label_rels = rels
        else:
            set_edges(rels, graph.matrices[code])
        relations += rels
    ctx = Context(
        sizes=[len(graph.ids[k]) for k in kind_names], relations=relations, label_rels=label_rels,
        label_shape=shape, out_kinds=(kind_index[kinds[0]], kind_index[kinds[1]]), kinds=kinds, gnn=gnn, ml=ml,
    )
    n_edges = sum(r.n_edges for r in relations if r not in label_rels)
    print(f"  {len(codes)} edge types -> {len(relations)} relations, {n_edges:,} directed messages "
          f"(+ label edge {label_code}: {'in graph, training pairs only' if label_rels else 'not in graph'})")
    print(f"  Positives: {len(fit_pos[0])} training, {len(val_pos[0])} validation ({gnn['validation_fold']}), "
          f"{len(test_pos[0])} test. Negatives: {len(neg_pool[0])} training pool, "
          f"{val_neg.sum():,} validation, {eval_neg.sum():,} test")
    return SimpleNamespace(ctx=ctx, graph=graph, kinds=kinds, codes=codes, out_dir=out_dir,
                           fit_pos=fit_pos, val_pos=val_pos, val_neg=val_neg, dev_pos=dev_pos,
                           test_pos=test_pos, eval_neg=eval_neg, neg_pool=neg_pool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to config file (default: config.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ml, gnn = cfg["ml"], cfg["gnn"]
    name, sel, ks = gnn["model_name"], ml["selection_metric"], ml["ranking_ks"]
    torch.manual_seed(gnn["seed"])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(gnn["seed"])
    t_start = time.time()
    d = prepare(cfg, rng)
    ctx, graph, kinds, out_dir = d.ctx, d.graph, d.kinds, d.out_dir
    test_pos, dev_pos, eval_neg, neg_pool = d.test_pos, d.dev_pos, d.eval_neg, d.neg_pool

    # --- model selection on the validation fold
    grid_rows, history, best = [], [], None
    points = grid(gnn["grid"])
    for g, params in enumerate(points):
        print(f"Grid point {g + 1}/{len(points)}: {params}")
        t = time.time()
        _, hist, res = fit(ctx, params, d.fit_pos, neg_pool, gnn["max_epochs"], f"grid{g}", (d.val_pos, d.val_neg))
        history += hist
        grid_rows.append({"params": params, "best_epoch": res["epoch"], "epochs_run": len(hist),
                          sel: res["metrics"][sel], "time_s": round(time.time() - t, 1)})
        print(f"  best val {sel} {res['metrics'][sel]:.4f} at epoch {res['epoch']} ({time.time() - t:.0f}s)")
        if best is None or res["metrics"][sel] > best["metrics"][sel]:
            best = {**res, "params": params, "grid_index": g}
    print(f"Selected {best['params']}, {best['epoch']} epochs (val {sel} {best['metrics'][sel]:.4f})")

    # --- test model: all development positives, selected epoch count
    print("Training the test model on all development positives ...")
    t = time.time()
    model, hist, _ = fit(ctx, best["params"], dev_pos, neg_pool, best["epoch"], "test_model")
    train_time = time.time() - t
    history += hist
    scores = predict(model, ctx, dev_pos)
    test = evaluate(scores, test_pos, eval_neg, ks, kinds)
    test_ci = bootstrap_ci(scores, test_pos, eval_neg, ml["bootstrap_samples"], rng)
    cold = ~np.isin(test_pos[0], dev_pos[0])
    subsets = {
        s: evaluate(scores, (test_pos[0][m], test_pos[1][m]), eval_neg, ks, kinds)
        for s, m in {"cold_drug": cold, "warm_drug": ~cold}.items() if m.any()
    }
    print(f"  {name}: {fmt(test)}")
    print(f"    AUROC 95% CI {test_ci['auroc_ci95']}, AUPRC 95% CI {test_ci['auprc_ci95']}; "
          + ", ".join(f"{s} AUPRC={v['auprc']:.3f}" for s, v in subsets.items()))

    # --- outputs (replace own entry / column only)
    entry = {
        "type": "rgcn",
        "params": {**best["params"], "num_bases": gnn["num_bases"], "decoder": gnn["decoder"],
                   "epochs": best["epoch"]},
        "validation": {
            "fold": gnn["validation_fold"],
            "note": ("Selected on a single held-out development fold, not 5-fold CV like the other "
                     "models; not compared for best_model."),
            "metrics": best["metrics"],
            "grid": grid_rows,
        },
        "test": test,
        "test_ci95": test_ci,
        "test_subsets": subsets,
        "training": {
            "epochs": best["epoch"],
            "test_model_train_time_s": round(train_time, 1),
            "total_time_s": round(time.time() - t_start, 1),
            "graph_edge_types": d.codes,
            "n_relations": len(ctx.relations),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "training_negatives": "sampled from dev_negatives.csv",
        },
        "note": "Test scores are raw decoder logits. Run after train.py, which overwrites this entry.",
    }
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["models"][name] = entry
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    frame = pd.read_csv(out_dir / "test_scores.csv.gz")   # re-read: keep other columns as they are now
    fr, fc = graph.lookup(kinds[0], frame["drug_id"]), graph.lookup(kinds[1], frame["disease_id"])
    frame[name] = scores[fr, fc]
    frame.to_csv(out_dir / "test_scores.csv.gz", index=False)
    pd.DataFrame(history).to_csv(out_dir / "gnn_history.csv", index=False)

    print(f"Done in {time.time() - t_start:.0f}s. Wrote {name} to {metrics_path}, test_scores.csv.gz "
          f"and gnn_history.csv")
    print("Next: python src/plots.py")


if __name__ == "__main__":
    main()
