from pathlib import Path

import ccxt
import pandas as pd


def main() -> None:
    exchange = ccxt.binance({"enableRateLimit": True})

    since = exchange.parse8601("2020-01-01T00:00:00Z")
    all_candles = []

    while True:
        candles = exchange.fetch_ohlcv(
            symbol="BTC/USDT",
            timeframe="1h",
            since=since,
            limit=1000,
        )

        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 60 * 60 * 1000

        print(f"Downloaded {len(all_candles):,} candles so far...")

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(all_candles, columns=columns)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    output_path = Path("data/raw/btcusdt_1h.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} candles to {output_path}")


if __name__ == "__main__":
    main()