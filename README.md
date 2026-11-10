# Drug Repurposing with a Knowledge Graph and ML

Predict which existing drugs might treat other diseases, using link prediction
on the [Hetionet](https://het.io) knowledge graph, with an Obsidian-style
interactive graph viewer.

## Setup

```powershell
pip install -r requirements.txt
python src/download_data.py      # ~15 MB into data/raw/
python src/build_subgraph.py     # writes web/data/graph.js
```

Then open `web/index.html` in a browser. It needs internet access to load the
`force-graph` library from unpkg.

## Viewer

- Hover a node to highlight its neighbors. Click it to open the side panel.
- Search for any drug, disease, gene or pathway by name.
- Use the legend checkboxes to hide node types.
- Gold rings mark **heuristic candidates**: drugs that bind several genes
  associated with a disease but are not known treatments for it.
- Model predictions show as gold dashed links once you add them (see below).

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
src/build_subgraph.py  extracts the subgraph and writes web/data/graph.js
web/                   viewer (index.html, app.js, style.css)
```
