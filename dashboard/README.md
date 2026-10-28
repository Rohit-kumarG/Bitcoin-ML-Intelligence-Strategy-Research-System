# Dashboard

This React screen uses the FastAPI research API and displays the live Binance BTC/USDT USD-M 1-hour candlestick chart. It polls the candle endpoint every 15 seconds and includes the forming candle.

The next-candle panel displays the classifier's UP/DOWN probabilities alongside a separate regression model's estimated next-hour open, high, low, and close. Those are out-of-sample evaluated estimates, not exact future prices or guarantees.

Before starting the API, create the OHLC regression artifact from the project root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m src.bitcoin_ml.models.train_futures_ohlc_regressor
```

## Start the API (PowerShell, from the project root)

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn src.bitcoin_ml.api.app:app --reload --host 127.0.0.1 --port 8000
```

The API loads the direction classifier and OHLC regressor once at startup. If the classifier is missing, create it with `python -m src.bitcoin_ml.models.train_futures_artifact`. Its dashboard data is at `http://127.0.0.1:8000/api/dashboard`, and the interactive API reference is at `http://127.0.0.1:8000/docs`.

## Start React (in a second PowerShell terminal)

```powershell
Set-Location -LiteralPath (Join-Path (Get-Location) 'dashboard')
npm.cmd install
npm.cmd run dev
```

Open the local URL Vite prints (normally `http://localhost:5173`). The dashboard refreshes data every minute and shows a one-second UTC clock.

## Candle-time meaning

Binance timestamps OHLCV rows at candle open. Once a completed candle closes, its row's past-only features are passed to the model. The response separately reports the completed input candle's open and close times and the following candle interval the estimate refers to. In-progress candles are ignored. Stale, gapped, or incomplete feature windows do not produce a direction estimate.
