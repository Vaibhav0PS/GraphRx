# Drug Repurposing with a Knowledge Graph and ML

Predict which existing drugs might treat other diseases, using link prediction
on the [Hetionet](https://het.io) knowledge graph, with an Obsidian-style
interactive graph viewer.

## Setup

```powershell
pip install -r requirements.txt
python src/download_data.py      # ~15 MB into data/raw/
python src/train.py              # ~1 min: CV, test evaluation, predictions
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
- **Evaluation negatives:** every unknown pair not used in training (~207k vs
  151 positives), so AUPRC reflects realistic class imbalance. Ranking metrics
  (MRR, Hits@K) are computed per drug and per disease. Results are also split
  by cold-start drugs (no training indication) and warm drugs.
- **Predictions:** the best model is retrained on all labels. The top 20
  unknown drugs per disease go to `results/predictions.csv`, with the
  metapaths that raised each score as evidence.

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
| `dev_negatives.csv` | Training negatives of the test-evaluation model (shared with other models) |
| `figures/` | ROC, PR, model comparison, feature importance |

Then open `web/index.html` in a browser. It needs internet access to load the
`force-graph` library from unpkg.

## Viewer

- Hover a node to highlight its neighbors. Click it to open the side panel.
- Search for any drug, disease, gene or pathway by name.
- Use the legend checkboxes to hide node types.
- Gold rings mark **heuristic candidates**: drugs that bind several genes
  associated with a disease but are not known treatments for it.
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
src/build_subgraph.py  extracts the subgraph and writes web/data/graph.js
web/                   viewer (index.html, app.js, style.css)
```
