"""One-time final holdout evaluation for the frozen Futures RF research setup."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)


FEATURES_PATH = Path("data/processed/btcusdt_usdm_1h_features.csv")
FUNDING_PATH = Path("data/raw/btcusdt_usdm_funding.csv")
PREDICTIONS_PATH = Path("artifacts/metrics/futures_final_test_predictions.csv")
MODEL_METRICS_PATH = Path("artifacts/metrics/futures_final_test_model_metrics.csv")
STRATEGY_PATH = Path("artifacts/metrics/futures_final_test_strategy_grid.csv")
FEATURES = [
    "return_1h", "return_3h", "return_6h", "return_24h", "volatility_24h",
    "volume_log_change_1h", "candle_body_pct", "candle_range_pct",
    "upper_wick_pct", "lower_wick_pct",
]
TARGET = "target_up"
FINAL_TEST_FRACTION = 0.15
THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
COST_SCENARIOS = [(0.001, 0.0005), (0.001, 0.001), (0.002, 0.002)]
INITIAL_CAPITAL = 10_000.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def main() -> None:
    for path in [FEATURES_PATH, FUNDING_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required final-evaluation input missing: {path}")

    data = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])
    data = data.dropna(subset=FEATURES + [TARGET, "next_return"]).sort_values(
        "timestamp"
    ).reset_index(drop=True)
    data[TARGET] = data[TARGET].astype(int)

    test_start = int(len(data) * (1 - FINAL_TEST_FRACTION))
    # Purge the final development row because its next-hour label overlaps the test boundary.
    train = data.iloc[: test_start - 1]
    test = data.iloc[test_start:].copy()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train[TARGET])
    test["actual_up"] = test[TARGET]
    test["probability_up"] = model.predict_proba(test[FEATURES])[:, 1]
    test["predicted_up"] = (test["probability_up"] >= 0.5).astype(int)
    model_metrics = pd.DataFrame(
        [
            {
                "train_rows": len(train),
                "purged_boundary_rows": 1,
                "test_rows": len(test),
                "train_end": train["timestamp"].max(),
                "test_start": test["timestamp"].min(),
                "test_end": test["timestamp"].max(),
                "accuracy": accuracy_score(test[TARGET], test["predicted_up"]),
                "balanced_accuracy": balanced_accuracy_score(
                    test[TARGET], test["predicted_up"]
                ),
                "precision": precision_score(test[TARGET], test["predicted_up"], zero_division=0),
                "recall": recall_score(test[TARGET], test["predicted_up"], zero_division=0),
                "brier_score": brier_score_loss(test[TARGET], test["probability_up"]),
                "log_loss": log_loss(test[TARGET], test["probability_up"], labels=[0, 1]),
            }
        ]
    )

    funding = pd.read_csv(FUNDING_PATH, parse_dates=["timestamp"])
    # Feature rows are timestamped at candle open; their one-hour forecast is
    # evaluated at the following close. Align settled funding to the held interval.
    funding["timestamp"] = funding["timestamp"] - pd.Timedelta(hours=2)
    test = test.merge(
        funding.rename(columns={"funding_rate": "settlement_funding_rate"})[
            ["timestamp", "settlement_funding_rate"]
        ],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    test["settlement_funding_rate"] = test["settlement_funding_rate"].fillna(0.0)

    elapsed_years = max(
        (test["timestamp"].iloc[-1] - test["timestamp"].iloc[0]).total_seconds()
        / (365.25 * 24 * 60 * 60),
        1 / (365.25 * 24),
    )
    summaries = []
    for fee_per_side, slippage_per_side in COST_SCENARIOS:
        cost = fee_per_side + slippage_per_side
        cost_name = f"fee_{fee_per_side:.4f}_slippage_{slippage_per_side:.4f}"
        benchmark_returns = test["next_return"].copy()
        benchmark_returns.iloc[0] = (1 + benchmark_returns.iloc[0]) * (1 - cost) - 1
        benchmark_returns.iloc[-1] = (1 + benchmark_returns.iloc[-1]) * (1 - cost) - 1
        benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns).cumprod()

        for threshold in THRESHOLDS:
            run = test.copy()
            run["position"] = 0
            run.loc[run["probability_up"] >= threshold, "position"] = 1
            run.loc[run["probability_up"] <= 1 - threshold, "position"] = -1
            previous = run["position"].shift(1, fill_value=0)
            change_units = (run["position"] - previous).abs()
            pnl = run["position"] * run["next_return"]
            funding_pnl = -run["position"] * run["settlement_funding_rate"]
            net_return = (1 + pnl + funding_pnl) * (1 - cost) ** change_units - 1
            final_exit_units = abs(int(run["position"].iloc[-1]))
            if final_exit_units:
                net_return.iloc[-1] = (1 + net_return.iloc[-1]) * (1 - cost) ** final_exit_units - 1
            equity = INITIAL_CAPITAL * (1 + net_return).cumprod()
            cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / elapsed_years) - 1
            active = run["position"] != 0

            summaries.append(
                {
                    "threshold": threshold,
                    "cost_scenario": cost_name,
                    "fee_per_side_pct": 100 * fee_per_side,
                    "slippage_per_side_pct": 100 * slippage_per_side,
                    "test_rows": len(run),
                    "long_hours": int((run["position"] == 1).sum()),
                    "short_hours": int((run["position"] == -1).sum()),
                    "time_in_market_pct": 100 * active.mean(),
                    "position_change_units_including_final_exit": float(
                        change_units.sum() + final_exit_units
                    ),
                    "ending_equity": equity.iloc[-1],
                    "total_return_pct": 100 * (equity.iloc[-1] / INITIAL_CAPITAL - 1),
                    "cagr_pct": 100 * cagr,
                    "max_drawdown_pct": 100 * max_drawdown(equity),
                    "buy_hold_ending_equity": benchmark_equity.iloc[-1],
                    "buy_hold_max_drawdown_pct": 100 * max_drawdown(benchmark_equity),
                    "funding_cashflow_pct_not_compounded": 100 * funding_pnl.sum(),
                }
            )

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    test[[
        "timestamp", "actual_up", "probability_up", "predicted_up", "next_return",
        "settlement_funding_rate",
    ]].to_csv(PREDICTIONS_PATH, index=False)
    model_metrics.to_csv(MODEL_METRICS_PATH, index=False)
    strategy = pd.DataFrame(summaries)
    strategy.to_csv(STRATEGY_PATH, index=False)

    print("ONE-TIME final holdout evaluation; do not tune from these results")
    print(f"Training rows: {len(train):,}; purged boundary rows: 1")
    print(f"Final test rows: {len(test):,}; period: {test['timestamp'].min()} to {test['timestamp'].max()}")
    print("\nModel prediction metrics:")
    print(model_metrics.round(4).to_string(index=False))
    print("\nPredeclared long/short thresholds and cost scenarios:")
    print(strategy[[
        "threshold", "fee_per_side_pct", "slippage_per_side_pct", "long_hours",
        "short_hours", "time_in_market_pct", "ending_equity", "total_return_pct",
        "max_drawdown_pct", "buy_hold_ending_equity",
    ]].round(3).to_string(index=False))
    print(f"\nSaved final test predictions to: {PREDICTIONS_PATH}")
    print(f"Saved model metrics to: {MODEL_METRICS_PATH}")
    print(f"Saved strategy grid to: {STRATEGY_PATH}")


if __name__ == "__main__":
    main()
