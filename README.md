# Bitcoin ML Intelligence & Strategy Research System

An educational, research-only system for studying BTC/USDT market data with machine learning. It includes historical spot and USDⓈ-M Futures ingestion, data-quality checks, past-only features, chronological validation, regime/anomaly analysis, backtesting with estimated costs and funding, drift monitoring, and a local SQLite experiment registry.

This repository does not promise predictive accuracy or profit and does not place live orders. Backtests are simplified historical simulations; they do not reproduce every exchange execution, margin, or liquidation detail.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Futures research workflow

Run commands from the repository root. The download uses public market data and does not need API keys.

```powershell
python .\src\bitcoin_ml\data\download_binance_usdm.py
$env:PYTHONPATH = (Get-Location).Path
python -m src.bitcoin_ml.features.run_futures_feature_pipeline
python .\src\bitcoin_ml\validation\walk_forward_futures.py
python .\src\bitcoin_ml\validation\backtest_futures_long_short.py
python .\src\bitcoin_ml\monitoring\monitor_futures_drift.py
python .\src\bitcoin_ml\tracking\register_futures_run.py
```

The one-time final holdout evaluation is separate:

```powershell
python .\src\bitcoin_ml\validation\final_futures_evaluation.py
```

Do not tune a model or threshold using the final holdout results. The final period is reserved for a one-time evaluation.

## Spot research

The original spot-data workflow and exploratory notebook remain available in `configs/`, `src/`, and `notebooks/`.

## Generated data and artifacts

Raw and processed datasets, model metrics, and the local experiment database are written under `data/` and `artifacts/`. Those generated files are ignored by Git; rerun the workflows above to recreate them.
