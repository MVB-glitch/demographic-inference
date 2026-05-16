# QueryChat Explorer

This repo includes a small QueryChat/Shiny explorer for the demographic
inference pipeline output. It is adapted from the social-listening explorer
pattern, but focused only on age, gender, location, metrics, DST evidence, SQL,
and per-user reasoning.

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r querychat_explorer/requirements.txt
```

Create or update `.env`:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
# Optional:
# QUERYCHAT_GEMINI_MODEL=gemini-2.5-flash
# QUERYCHAT_ENABLE_VIZ=false
```

## Generate Pipeline JSON

Run the inference pipeline with JSON output:

```powershell
demoinfer --csv data/sample_users.csv --format json --output results.json
```

For UI testing without calling the LLM, generate synthetic output:

```powershell
python querychat_explorer/generate_synthetic_results.py --users 100 --output results.json
```

Or use the committed 100-user synthetic example:

```powershell
python querychat_explorer/prepare_querychat_data.py querychat_explorer/examples/results.json
```

JSON exports now use this structured shape:

- `metadata`
- `results`
- `metrics`
- `metric_rows`
- `confusion_matrices`
- `confusion_rows`
- `distributions`

## Prepare QueryChat Files

```powershell
python querychat_explorer/prepare_querychat_data.py results.json
```

This writes:

- `outputs/querychat/inference_results.csv`
- `outputs/querychat/metric_summary.csv`
- `outputs/querychat/confusion_matrix.csv`
- `outputs/querychat/distributions.csv`
- `outputs/querychat/metadata.json`
- `outputs/querychat/structured_inference_output.json`
- `outputs/querychat/demographic_inference.duckdb`

The prep script also accepts older pipeline JSON files that are plain lists of
result records.

## Run

```powershell
shiny run --reload querychat_explorer/app.py
```

The app includes:

- QueryChat sidebar for text-to-SQL exploration
- KPI cards for filtered results
- prediction distribution charts
- accuracy, precision, recall, F1, and age MAE
- confusion matrices for location, age, and gender
- DST source-evidence and final belief summaries
- filtered user table
- reasoning previews
- current SQL panel
