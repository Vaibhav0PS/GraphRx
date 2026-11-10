# Drug Repurposing with a Knowledge Graph and ML

Predict which existing drugs might treat other diseases, using link prediction
on the [Hetionet](https://het.io) knowledge graph, with an Obsidian-style
interactive graph viewer.

## Setup

```powershell
pip install -r requirements.txt
python src/download_data.py      # ~15 MB into data/raw/
python src/train.py              # ~1.5 min: CV, test evaluation, predictions
python src/train_gnn.py          # ~5 min: R-GCN, same test set (run after train.py)
python src/plots.py              # figures in results/figures/
python src/explain.py            # concrete graph paths behind each prediction
python src/build_subgraph.py     # writes web/data/graph.js (with predictions)
```

## ML method

Task: predict missing `Compound -treats-> Disease` edges (link prediction).

- **Features:** degree-weighted path counts (DWPC) for the metapaths in
  `config.json`, e.g. `CbGaD` = drug binds a gene that is associated with the
  disease, `CrCtD` = a similar drug treats the disease. Walks that step out to a
  node and straight back are dropped. Degree features capture popularity.
- **No leakage:** test labels are removed from the graph before any feature is
  computed. Training positives get out-of-fold features, so a pair's own
  treats edge never contributes to its own features.
- **Negatives:** random drug-disease pairs with no known treats or palliates
  edge (10 per positive). Unknown is not the same as false, so this is a
  positive-unlabeled setting.
- **Models:** degree baseline (popularity only), logistic regression and
  XGBoost. Hyperparameters are chosen by 5-fold cross-validation, and each
  model is evaluated once on a 20% held-out test set.
- **Test negatives:** half of all unknown pairs (~106k vs 151 positives) are
  reserved before any training or model selection. CV and the GNN's
  validation only use the other half, so test negatives never influence which
  model or settings are picked. The class balance stays realistic for AUPRC.
  Ranking metrics (MRR, Hits@K) are computed per drug and per disease.
- **Cold-start vs warm drugs:** results are also reported for test drugs with
  no development treatment edge (cold start) and the rest (warm). Each cohort's
  positives and negatives involve only that cohort's drugs, and its positive
  rate is reported, because AUPRC depends on it.
- **R-GCN (`src/train_gnn.py`):** a relational graph neural network in plain
  PyTorch (no PyG) over all 24 Hetionet edge types, with basis-decomposed
  relation weights and a DistMult decoder. Test treats edges are never in its
  graph. Each epoch, training positives are split into message-passing edges
  and supervised edges, so no pair is predicted through its own edge. It uses
  the same training negatives (`dev_negatives.csv`) and the same reserved test
  pairs as the other models. Hyperparameters and epoch count are picked on one
  validation fold (`cv0`), not 5-fold CV. `train.py` overwrites its results,
  so run it after `train.py`.
- **Predictions:** the best model is retrained on all labels. The top 20
  unknown drugs per disease go to `results/predictions.csv`, with the
  metapaths that raised each score as evidence.

### Results (test set: 151 positives vs 105,740 reserved negatives)

| Model | AUROC | AUPRC (95% CI) | Hits@10 per drug | Hits@10 per disease | Cold-start AUPRC | Warm AUPRC |
|---|---|---|---|---|---|---|
| Degree baseline | 0.768 | 0.032 (0.022–0.049) | 0.30 | 0.28 | 0.001 | 0.051 |
| Logistic regression | 0.972 | 0.287 (0.225–0.359) | 0.93 | 0.62 | 0.237 | 0.357 |
| XGBoost (selected by CV) | 0.976 | 0.311 (0.255–0.382) | 0.93 | 0.68 | 0.240 | 0.404 |
| R-GCN | 0.904 | 0.032 (0.023–0.052) | 0.60 | 0.25 | 0.006 | 0.047 |

A random ranking would get AUPRC 0.0014 overall, 0.0007 for cold-start drugs
(59 positives, 83,296 negatives) and 0.0041 for warm drugs (92 positives,
22,444 negatives). Numbers are from a clean install of `requirements.txt`.
XGBoost results shift slightly between versions: 3.1.1 gave test AUPRC 0.337,
3.4.1 gave 0.311. The other models are identical across versions.

- **Metapath models vs R-GCN:** the metapath models win by a wide margin, and
  the GNN's validation score shows the same gap, so this is not test-set
  noise. Likely reasons: only ~480 training positives for a model with ~3M
  parameters (it overfits within ~80 epochs), validation scores that vary a
  lot across folds and seeds, and metapath features that encode exactly the
  drug-gene-disease patterns the task depends on.
- **Cold start:** for drugs with no development treatment edge, the metapath
  models still reach AUPRC ~0.24, about 340 times the random rate, because
  features such as shared targets and side effects don't need a known
  indication. The degree baseline and the R-GCN fail almost completely on
  these drugs; both rely on the drug's own treatment edges.

### Outputs (`results/`)

| File | Contents |
|---|---|
| `metrics.json` | CV score, test metrics, bootstrap 95% CI, cold/warm split per model |
| `cv_results.csv` | Score of every grid point on every CV fold |
| `test_scores.csv.gz` | Test pairs with labels and each model's score |
| `feature_importance.csv` | LR coefficients and XGBoost gain |
| `split.csv` | Which positives went to test and to which CV fold |
| `predictions.csv` | Repurposing candidates with evidence |
| `explanations.json` | Top concrete paths per prediction and evidence metapath |
| `prediction_graph_label_edges.csv` | Treats edges in the graph that scored the predictions (explanations use the same graph) |
| `dev_negatives.csv` | Training negatives of the test-evaluation model (shared with other models) |
| `gnn_history.csv` | R-GCN loss and validation AUPRC per epoch for every run |
| `figures/` | ROC, PR, model comparison, feature importance |

Then open `web/index.html` in a browser. It needs internet access to load the
`force-graph` library from unpkg.

## Viewer

- Hover a node to highlight its neighbors. Click it to open the side panel.
- Search for any drug, disease, gene or pathway by name.
- Use the legend checkboxes to hide node types.
- Gold rings mark **heuristic candidates**: drugs that bind several genes
  associated with a disease but are not known treatments for it.
- With nothing selected, the side panel lists the **top predictions** grouped
  by disease. Click a drug to open it, or press Escape on the list to hide the panel.
- Gold is used only for predictions: dashed gold links are predicted
  treatments, a solid gold ring marks a drug with a prediction, and a dotted
  ring marks a heuristic candidate.
- Model predictions show as gold dashed links once you add them (see below).
- Select a drug or disease with predictions to see **Why predicted?**: the
  strongest concrete paths behind each prediction, grouped by metapath, with
  the metapath's contribution to the score. Click a path to highlight it in
  the graph. Escape clears it.

## Configuration (`config.json`)

| Section | What it controls |
|---|---|
| `data` | Download URL and file names |
| `schema` | Which Hetionet edge types play each role, written as `[source kind, relation, target kind]`. Abbreviations are looked up in the Hetionet metagraph. |
| `subgraph` | Seed diseases (an empty list auto-picks the `auto_seed_count` diseases with the most known treatments), candidate count, pathway limits, output folder |
| `predictions` | CSV file and column names for model scores to overlay |

Command-line options override the config:

```powershell
python src/build_subgraph.py --list-diseases
python src/build_subgraph.py --diseases asthma psoriasis "Alzheimer's disease"
python src/build_subgraph.py --predictions results/predictions.csv
```

## Project layout

```
config.json            all settings
src/config.py          config loading and paths
src/hetionet.py        data loading, edge-type lookup from the metagraph
src/download_data.py   downloads the data
src/features.py        metapath parsing and DWPC feature matrices
src/train.py           split, CV, models, evaluation, predictions
src/plots.py           result figures
src/explain.py         path explanations for predictions
src/train_gnn.py       R-GCN model evaluated on the same split
src/build_subgraph.py  extracts the subgraph and writes web/data/graph.js
web/                   viewer (index.html, app.js, style.css)
```
