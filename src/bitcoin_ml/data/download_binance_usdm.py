"""Download public Binance USDⓈ-M BTCUSDT perpetual candles and funding history."""

from pathlib import Path
import time

import ccxt
import pandas as pd


SYMBOL = "BTC/USDT:USDT"
TIMEFRAME = "1h"
START_DATE = "2020-01-01T00:00:00Z"
CANDLE_PATH = Path("data/raw/btcusdt_usdm_1h.csv")
FUNDING_PATH = Path("data/raw/btcusdt_usdm_funding.csv")
PAGE_LIMIT = 1000
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 3


def fetch_with_retry(fetch_page, *args, **kwargs):
    """Retry temporary network/exchange errors without losing saved batches."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_page(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as error:
            if attempt == MAX_RETRIES:
                raise
            delay = RETRY_DELAY_SECONDS * attempt
            print(
                f"Temporary Binance connection error ({attempt}/{MAX_RETRIES}); "
                f"retrying in {delay}s: {error}"
            )
            time.sleep(delay)


def append_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def download_candles(exchange: ccxt.Exchange, since: int) -> pd.DataFrame:
    if CANDLE_PATH.exists():
        saved = pd.read_csv(CANDLE_PATH, usecols=["timestamp"], parse_dates=["timestamp"])
        if not saved.empty:
            since = int(saved["timestamp"].max().timestamp() * 1000) + 60 * 60 * 1000
            print(f"Resuming futures candles from {pd.to_datetime(since, unit='ms', utc=True)}")

    current_hour_ms = int(pd.Timestamp.now(tz="UTC").floor("h").timestamp() * 1000)
    while True:
        page = fetch_with_retry(
            exchange.fetch_ohlcv,
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since,
            limit=PAGE_LIMIT,
        )
        if not page:
            break

        completed = [row for row in page if row[0] < current_hour_ms]
        if completed:
            frame = pd.DataFrame(
                completed,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            append_checkpoint(frame, CANDLE_PATH)

        if page[-1][0] >= current_hour_ms or len(page) < PAGE_LIMIT:
            break
        next_since = page[-1][0] + 60 * 60 * 1000
        if next_since <= since:
            raise RuntimeError("Candle download cursor stopped advancing.")
        since = next_since
        print(f"Saved futures candle checkpoint through {frame['timestamp'].max()}...")

    if not CANDLE_PATH.exists():
        raise RuntimeError("Binance returned no completed futures candles.")
    frame = pd.read_csv(CANDLE_PATH, parse_dates=["timestamp"])
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    frame.to_csv(CANDLE_PATH, index=False)
    return frame


def download_funding(exchange: ccxt.Exchange, since: int) -> pd.DataFrame:
    if not exchange.has.get("fetchFundingRateHistory"):
        raise RuntimeError("This CCXT version does not support funding-rate history.")

    if FUNDING_PATH.exists():
        saved = pd.read_csv(FUNDING_PATH, usecols=["timestamp"], parse_dates=["timestamp"])
        if not saved.empty:
            since = int(saved["timestamp"].max().timestamp() * 1000) + 1
            print(f"Resuming funding history from {pd.to_datetime(since, unit='ms', utc=True)}")

    while True:
        page = fetch_with_retry(
            exchange.fetch_funding_rate_history,
            SYMBOL,
            since=since,
            limit=PAGE_LIMIT,
        )
        if not page:
            break
        page_timestamps = [item["timestamp"] for item in page if item.get("timestamp")]
        if not page_timestamps:
            raise RuntimeError("Funding response did not include timestamps.")
        checkpoint = pd.DataFrame(
            {
                "timestamp": [item["timestamp"] for item in page],
                "funding_rate": [item["fundingRate"] for item in page],
            }
        )
        checkpoint["timestamp"] = pd.to_datetime(
            checkpoint["timestamp"], unit="ms", utc=True
        )
        append_checkpoint(checkpoint, FUNDING_PATH)
        next_since = max(page_timestamps) + 1
        if next_since <= since:
            raise RuntimeError("Funding download cursor stopped advancing.")
        since = next_since
        print(f"Saved funding checkpoint through {checkpoint['timestamp'].max()}...")
        if len(page) < PAGE_LIMIT:
            break

    if not FUNDING_PATH.exists():
        raise RuntimeError("Binance returned no funding-rate records.")
    frame = pd.read_csv(FUNDING_PATH, parse_dates=["timestamp"])
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    frame.to_csv(FUNDING_PATH, index=False)
    return frame


def main() -> None:
    exchange = ccxt.binanceusdm({"enableRateLimit": True, "timeout": 60_000})
    since = exchange.parse8601(START_DATE)

    candles = download_candles(exchange, since)
    funding = download_funding(exchange, since)

    CANDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FUNDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    candles.to_csv(CANDLE_PATH, index=False)
    funding.to_csv(FUNDING_PATH, index=False)

    print(f"\nSaved {len(candles):,} completed futures candles to {CANDLE_PATH}")
    print(f"Candle range: {candles['timestamp'].min()} to {candles['timestamp'].max()}")
    print(f"Saved {len(funding):,} funding records to {FUNDING_PATH}")
    print(f"Funding range: {funding['timestamp'].min()} to {funding['timestamp'].max()}")


if __name__ == "__main__":
    main()
