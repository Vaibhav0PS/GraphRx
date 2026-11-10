"""Train and evaluate drug-disease link prediction models on Hetionet.

Pipeline (all settings in the "ml" section of config.json):

1. Hold out `test_fraction` of the known label edges (e.g. Compound-treats-
   Disease) as the test set. The rest is the development set.
2. Training set construction (used for CV folds, the test model and the final
   model alike): given some known positives, sample negatives from pairs with
   no known label edge. Features are computed OUT-OF-FOLD: rows are split into
   `oof_folds` groups and each group's features come from a graph without
   that group's label edges. So no training positive ever sees its own edge,
   not even through degree counts. Held-out pairs are scored on the graph
   with all training positives.
3. Model selection: for each model, every grid point is scored by
   `cv_folds`-fold cross-validation over the development positives.
4. Test: each model is refit on all development positives with its best
   parameters and evaluated once on the test positives. Negatives are every
   unknown pair not used for training, so positives are rare, as in reality.
   AUROC/AUPRC come with a bootstrap 95% CI, and results are also reported
   separately for test drugs with and without any training indication.
5. The best model (by CV) is refit on all known labels and scores every
   unknown pair. The top pairs per disease are written as repurposing
   predictions, with the metapaths that pushed each score up as evidence.

Convention: negatives are sampled from pairs with no known label edge in ANY
split (the usual "filtered" setting), so held-out positives are never used as
negatives.

Outputs in ml.results_dir:
    metrics.json, cv_results.csv, test_scores.csv.gz, feature_importance.csv,
    split.csv, and the predictions file named in config predictions.file

Usage:
    python src/train.py [--config CONFIG]
"""

import argparse
import itertools
import json
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hetionet
from config import load_config, raw_files, resolve
from features import FeatureBuilder, Graph


# ---------------------------------------------------------------- models

class FeatureProduct:
    """Baseline: score = product of the given feature columns, e.g. drug and disease popularity."""

    def __init__(self, features: list[str]):
        self.features = list(features)

    def fit(self, X: pd.DataFrame, y) -> "FeatureProduct":
        missing = set(self.features) - set(X.columns)
        if missing:
            sys.exit(f"feature_product baseline needs features {sorted(missing)}; add them to degree_features.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = X[self.features].prod(axis=1).to_numpy()
        return np.column_stack([-s, s])


def make_model(kind: str, params: dict, seed: int):
    if kind == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed, **params)),
        ])
    if kind == "xgboost":
        return xgb.XGBClassifier(**params, random_state=seed, n_jobs=-1, tree_method="hist", eval_metric="aucpr")
    if kind == "feature_product":
        return FeatureProduct(**params)
    sys.exit(f"Unknown model type '{kind}'. Use logistic_regression, xgboost or feature_product.")


def importance(model, kind: str, names: list[str]) -> list[float] | None:
    if kind == "logistic_regression":
        return model.named_steps["clf"].coef_[0].tolist()
    if kind == "xgboost":
        gain = model.get_booster().get_score(importance_type="gain")
        return [gain.get(n, 0.0) for n in names]
    return None


def contributions(model, kind: str, X: pd.DataFrame) -> np.ndarray | None:
    """Per-pair, per-feature contribution to the score in log-odds, if the model has one."""
    if kind == "logistic_regression":
        return model.named_steps["scale"].transform(X) * model.named_steps["clf"].coef_[0]
    if kind == "xgboost":
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]  # drop bias column
    return None


def grid(spec: dict) -> list[dict]:
    keys = list(spec)
    values = [v if isinstance(v, list) else [v] for v in spec.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------- metrics

def ranking_metrics(scores: np.ndarray, rows, cols, neg_mask: np.ndarray, ks: list[int], suffix: str) -> dict:
    """Filtered ranking: each true pair is ranked against the evaluation negatives
    in the same row. Ties count as half."""
    ranks = []
    for r, c in zip(rows, cols):
        row, s = scores[r], scores[r, c]
        negs = row[neg_mask[r]]
        ranks.append(1 + np.count_nonzero(negs > s) + np.count_nonzero(negs == s) / 2)
    ranks = np.array(ranks)
    out = {f"mrr_{suffix}": float(np.mean(1 / ranks))}
    out.update({f"hits@{k}_{suffix}": float(np.mean(ranks <= k)) for k in ks})
    return out


def evaluate(scores: np.ndarray, pos: tuple, neg_mask: np.ndarray, ks, kinds) -> dict:
    """AUROC/AUPRC and ranking metrics, all against the same negative set."""
    pr, pc = pos
    neg = scores[neg_mask]
    y = np.r_[np.ones(len(pr)), np.zeros(len(neg))]
    s = np.r_[scores[pr, pc], neg]
    out = {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
    out.update(ranking_metrics(scores, pr, pc, neg_mask, ks, kinds[0]))
    out.update(ranking_metrics(scores.T, pc, pr, neg_mask.T, ks, kinds[1]))
    out.update({"n_positive": int(len(pr)), "n_negative": int(len(neg))})
    return out


def bootstrap_ci(scores: np.ndarray, pos: tuple, neg_mask: np.ndarray, n: int, rng) -> dict:
    """95% CI for AUROC/AUPRC by resampling the positives (the scarce, noisy part).
    Negatives are fixed, which allows exact, fast formulas via sorted negatives."""
    neg = np.sort(scores[neg_mask])
    pos_s = scores[pos]

    def auroc(p):
        below = np.searchsorted(neg, p, "left")
        ties = np.searchsorted(neg, p, "right") - below
        return np.mean((below + ties / 2) / len(neg))

    def auprc(p):
        p = np.sort(p)[::-1]
        k = np.arange(1, len(p) + 1)
        neg_above = len(neg) - np.searchsorted(neg, p, "left")
        return np.mean(k / (k + neg_above))

    draws = [rng.choice(pos_s, size=len(pos_s), replace=True) for _ in range(n)]
    return {
        "auroc_ci95": np.percentile([auroc(d) for d in draws], [2.5, 97.5]).round(4).tolist(),
        "auprc_ci95": np.percentile([auprc(d) for d in draws], [2.5, 97.5]).round(4).tolist(),
    }


# ---------------------------------------------------------------- data helpers

def pair_mask(graph: Graph, edges: pd.DataFrame, code: str, kinds: tuple, shape: tuple) -> np.ndarray:
    sub = edges[edges["metaedge"] == code]
    mask = np.zeros(shape, dtype=bool)
    mask[graph.lookup(kinds[0], sub["source"]), graph.lookup(kinds[1], sub["target"])] = True
    return mask


def sample(mask: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    flat = np.flatnonzero(mask)
    if n > len(flat):
        sys.exit(f"Asked for {n} negatives but only {len(flat)} candidate pairs exist.")
    return np.unravel_index(rng.choice(flat, size=n, replace=False), mask.shape)


class TrainingSet:
    """Out-of-fold training features for the given positives plus sampled negatives.

    `mats` are features from the graph with ALL the given positives; use them
    to score held-out pairs. `neg_mask` marks the sampled negatives, which
    must be left out of evaluation.
    """

    def __init__(self, builder: FeatureBuilder, pos: tuple, neg_pool: np.ndarray, ml: dict, rng):
        pr, pc = pos
        n_neg = ml["negatives_per_positive"] * len(pr)
        nr, nc = sample(neg_pool, n_neg, rng)
        self.neg_mask = np.zeros(neg_pool.shape, dtype=bool)
        self.neg_mask[nr, nc] = True

        self.mats, scales = builder.matrices(pr, pc)
        k = ml["split"]["oof_folds"]
        pos_fold = rng.permutation(len(pr)) % k
        neg_fold = rng.permutation(n_neg) % k
        parts = []
        for f in range(k):
            keep = pos_fold != f
            mats_f, _ = builder.matrices(pr[keep], pc[keep], scales=scales)
            p, n = ~keep, neg_fold == f
            X = builder.frame(mats_f, np.r_[pr[p], nr[n]], np.r_[pc[p], nc[n]])
            X["_y"] = np.r_[np.ones(p.sum()), np.zeros(n.sum())]
            parts.append(X)
        data = pd.concat(parts, ignore_index=True)
        self.y = data.pop("_y").to_numpy()
        self.X = data


def score_all(model, builder: FeatureBuilder, mats: dict, shape: tuple) -> np.ndarray:
    rows, cols = np.indices(shape)
    X = builder.frame(mats, rows.ravel(), cols.ravel())
    return model.predict_proba(X)[:, 1].reshape(shape)


def fmt(metrics: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float))


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to config file (default: config.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ml = cfg["ml"]
    rng = np.random.default_rng(ml["seed"])
    out_dir = resolve(ml["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    metagraph = hetionet.Metagraph(raw_files(cfg)["metagraph"])
    nodes, edges, metagraph = hetionet.load(cfg, codes=FeatureBuilder.required_codes(ml, metagraph))
    graph = Graph(nodes, edges, metagraph)
    builder = FeatureBuilder(graph, ml)
    kinds = (builder.drug_kind, builder.disease_kind)
    shape = (len(graph.ids[kinds[0]]), len(graph.ids[kinds[1]]))
    print(f"  {shape[0]} {kinds[0]} x {shape[1]} {kinds[1]} = {shape[0] * shape[1]:,} candidate pairs, "
          f"{len(builder.feature_names)} features")

    # --- labels
    positive = pair_mask(graph, edges, builder.label_code, kinds, shape)
    excluded = np.zeros(shape, dtype=bool)
    for spec in ml["exclude_from_negatives"]:
        excluded |= pair_mask(graph, edges, metagraph.abbrev(spec), kinds, shape)
    excluded &= ~positive
    unknown = ~(positive | excluded)

    # --- test / development split, then CV folds over development
    pos_r, pos_c = np.nonzero(positive)
    order = rng.permutation(len(pos_r))
    n_test = round(len(order) * ml["split"]["test_fraction"])
    test_idx, dev_idx = order[:n_test], order[n_test:]
    n_cv = ml["split"]["cv_folds"]
    cv_fold = np.arange(len(dev_idx)) % n_cv
    print(f"  Label {builder.label_code}: {len(order)} positives, {len(dev_idx)} development "
          f"({n_cv}-fold CV), {n_test} test; {excluded.sum()} pairs excluded from negatives")

    pd.DataFrame({
        "drug_id": graph.ids[kinds[0]][pos_r[order]],
        "disease_id": graph.ids[kinds[1]][pos_c[order]],
        "split": ["test"] * n_test + [f"cv{f}" for f in cv_fold],
    }).to_csv(out_dir / "split.csv", index=False)

    sel = ml["selection_metric"]
    models = ml["models"]
    grids = {name: grid(spec.get("grid", {})) for name, spec in models.items()}

    # --- model selection by cross-validation
    print(f"Cross-validation ({n_cv} folds, {ml['split']['oof_folds']} out-of-fold feature groups each) ...")
    cv_rows = []
    for f in range(n_cv):
        tr, ho = dev_idx[cv_fold != f], dev_idx[cv_fold == f]
        ts = TrainingSet(builder, (pos_r[tr], pos_c[tr]), unknown, ml, rng)
        eval_neg = unknown & ~ts.neg_mask
        for name, spec in models.items():
            for g, params in enumerate(grids[name]):
                model = make_model(spec["type"], params, ml["seed"]).fit(ts.X, ts.y)
                scores = score_all(model, builder, ts.mats, shape)
                m = evaluate(scores, (pos_r[ho], pos_c[ho]), eval_neg, ml["ranking_ks"], kinds)
                cv_rows.append({"model": name, "grid_index": g, "params": json.dumps(params), "fold": f, sel: m[sel]})
        print(f"  fold {f + 1}/{n_cv} done")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out_dir / "cv_results.csv", index=False)
    summary = cv.groupby(["model", "grid_index", "params"], sort=False)[sel].agg(["mean", "std"]).reset_index()
    for row in summary.itertuples():
        print(f"  {row.model} {row.params}: CV {sel} = {row.mean:.4f} ± {row.std:.4f}")
    best_rows = summary.loc[summary.groupby("model", sort=False)["mean"].idxmax()]
    best_params = {r.model: grids[r.model][r.grid_index] for r in best_rows.itertuples()}
    cv_best = {r.model: {"mean": float(r.mean), "std": float(r.std)} for r in best_rows.itertuples()}
    best_name = max(cv_best, key=lambda n: cv_best[n]["mean"])
    print(f"Best model by CV {sel}: {best_name} {best_params[best_name]}")

    # --- test evaluation: refit on all development positives
    print("Evaluating on the test set ...")
    dev_ts = TrainingSet(builder, (pos_r[dev_idx], pos_c[dev_idx]), unknown, ml, rng)
    eval_neg = unknown & ~dev_ts.neg_mask
    test_pos = (pos_r[test_idx], pos_c[test_idx])
    dev_drugs = set(pos_r[dev_idx])
    cold = np.array([r not in dev_drugs for r in test_pos[0]])
    subsets = {"cold_drug": cold, "warm_drug": ~cold}
    print(f"  {cold.sum()} of {n_test} test pairs have a {kinds[0]} with no development label edge (cold start)")

    results, test_scores, imp_rows = {}, {}, []
    for name, spec in models.items():
        model = make_model(spec["type"], best_params[name], ml["seed"]).fit(dev_ts.X, dev_ts.y)
        scores = score_all(model, builder, dev_ts.mats, shape)
        test = evaluate(scores, test_pos, eval_neg, ml["ranking_ks"], kinds)
        test_ci = bootstrap_ci(scores, test_pos, eval_neg, ml["bootstrap_samples"], rng)
        by_subset = {
            s: evaluate(scores, (test_pos[0][m], test_pos[1][m]), eval_neg, ml["ranking_ks"], kinds)
            for s, m in subsets.items() if m.any()
        }
        results[name] = {
            "type": spec["type"], "params": best_params[name], "cv": {sel: cv_best[name]},
            "test": test, "test_ci95": test_ci, "test_subsets": by_subset,
        }
        test_scores[name] = scores
        imp = importance(model, spec["type"], builder.feature_names)
        if imp is not None:
            imp_rows += [{"model": name, "feature": f, "importance": v} for f, v in zip(builder.feature_names, imp)]
        print(f"  {name}: {fmt(test)}")
        print(f"    AUROC 95% CI {test_ci['auroc_ci95']}, AUPRC 95% CI {test_ci['auprc_ci95']}; "
              + ", ".join(f"{s} AUPRC={v['auprc']:.3f}" for s, v in by_subset.items()))

    metrics = {
        "label_edge": ml["label_edge"],
        "best_model": best_name,
        "selection_metric": sel,
        "split": {"development": int(len(dev_idx)), "test": int(n_test), "cv_folds": n_cv,
                  "test_cold_drug": int(cold.sum())},
        "negatives_per_positive": ml["negatives_per_positive"],
        "features": builder.feature_names,
        "models": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(imp_rows, columns=["model", "feature", "importance"]).to_csv(
        out_dir / "feature_importance.csv", index=False)

    en_r, en_c = np.nonzero(eval_neg)
    r, c = np.r_[test_pos[0], en_r], np.r_[test_pos[1], en_c]
    frame = pd.DataFrame({
        "drug_id": graph.ids[kinds[0]][r],
        "disease_id": graph.ids[kinds[1]][c],
        "label": np.r_[np.ones(n_test, dtype=int), np.zeros(len(en_r), dtype=int)],
    })
    for name, s in test_scores.items():
        frame[name] = s[r, c]
    frame.to_csv(out_dir / "test_scores.csv.gz", index=False)

    # --- final model on all labels
    if ml["final"]["retrain_on_all_labels"]:
        print("Retraining best model on all labels ...")
        ts = TrainingSet(builder, (pos_r, pos_c), unknown, ml, rng)
    else:
        ts = dev_ts
    spec = models[best_name]
    model = make_model(spec["type"], best_params[best_name], ml["seed"]).fit(ts.X, ts.y)
    scores = score_all(model, builder, ts.mats, shape)
    write_predictions(cfg, graph, builder, ts.mats, model, spec["type"], scores, unknown, kinds, nodes)

    print(f"Done in {time.time() - t0:.0f}s. Results in {out_dir}")
    print("Next: python src/plots.py   then   python src/build_subgraph.py")


def write_predictions(cfg, graph, builder, mats, model, kind, scores, unknown, kinds, nodes) -> None:
    """Top unknown pairs per disease. Evidence = metapaths that exist for the pair
    and raised its score, strongest first."""
    final, pcfg = cfg["ml"]["final"], cfg["predictions"]
    name_of = dict(zip(nodes["id"], nodes["name"]))

    rows_r, rows_c = [], []
    for c in range(scores.shape[1]):
        cand = np.flatnonzero(unknown[:, c])
        top = cand[np.argsort(-scores[cand, c], kind="stable")][: final["top_per_disease"]]
        rows_r += top.tolist()
        rows_c += [c] * len(top)
    rows_r, rows_c = np.array(rows_r), np.array(rows_c)

    X = builder.frame(mats, rows_r, rows_c)
    contrib = contributions(model, kind, X)
    names = builder.feature_names
    dwpc = [i for i, n in enumerate(names) if n.startswith("dwpc_")]

    out = []
    for i, (r, c) in enumerate(zip(rows_r, rows_c)):
        evidence = []
        if contrib is not None:
            evidence = sorted(
                ((contrib[i, j], names[j]) for j in dwpc if X.iat[i, j] > 0 and contrib[i, j] > 0), reverse=True
            )[: final["max_evidence"]]
        drug, disease = graph.ids[kinds[0]][r], graph.ids[kinds[1]][c]
        out.append({
            pcfg["drug_column"]: drug,
            "drug_name": name_of[drug],
            pcfg["disease_column"]: disease,
            "disease_name": name_of[disease],
            pcfg["score_column"]: round(float(scores[r, c]), 6),
            "rank_in_disease": int(np.sum(rows_c[:i] == c)) + 1,
            "evidence": "; ".join(f"{n.removeprefix('dwpc_')} (+{v:.2f})" for v, n in evidence),
        })
    path = resolve(pcfg["file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"  Wrote {len(out)} predictions to {path}")


if __name__ == "__main__":
    main()
