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

## Advanced ensemble workflow (maximum accuracy)

The advanced pipeline adds 50+ features (technical indicators, funding-rate signals, multi-timeframe trends, cyclical time encoding) and trains a four-model ensemble (LightGBM + XGBoost + CatBoost + Random Forest) with Optuna hyperparameter tuning. High-confidence thresholding further filters predictions for maximum directional accuracy.

```powershell
python .\src\bitcoin_ml\data\download_binance_usdm.py
$env:PYTHONPATH = (Get-Location).Path
python -m src.bitcoin_ml.features.run_advanced_feature_pipeline
python .\src\bitcoin_ml\validation\walk_forward_advanced.py
python .\src\bitcoin_ml\validation\final_advanced_evaluation.py
python -m src.bitcoin_ml.models.train_advanced_ensemble
```

The walk-forward script reports per-model and ensemble accuracy across chronological folds, plus a confidence-threshold analysis showing accuracy at various coverage levels. The final evaluation is a one-time holdout test.

## Spot research

The original spot-data workflow and exploratory notebook remain available in `configs/`, `src/`, and `notebooks/`.

## Local dashboard

Train the separate OHLC regression artifact before starting the API:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m src.bitcoin_ml.models.train_futures_ohlc_regressor
```

This model estimates next-hour open, high, low, and close returns. The workflow selects between a no-change baseline and a Random Forest using chronological validation, reports final chronological test errors, and only then refits the production artifact. Price outputs remain uncertain estimates, not exact future prices.

Start the FastAPI service and the React dashboard in separate PowerShell terminals. The dashboard shows the live estimate, exact UTC candle interval, data freshness, historical test metrics, backtest scenarios, monitoring data, and recorded live prediction outcomes. Its BTC candlestick chart polls Binance USD-M hourly candles, including the forming candle. The next-candle price box displays the separate OHLC regression model's estimated candle and direction-classifier probabilities side by side.

See [dashboard/README.md](dashboard/README.md) for commands. The API loads the saved Futures model locally and fetches only completed public Binance Futures candles; it does not place orders.

## Generated data and artifacts

Raw and processed datasets, model metrics, and the local experiment database are written under `data/` and `artifacts/`. Those generated files are ignored by Git; rerun the workflows above to recreate them.
