"""Simulate a simple long/flat strategy using out-of-fold probabilities only."""

from pathlib import Path

import pandas as pd


PROBABILITIES_PATH = Path("artifacts/metrics/walk_forward_probabilities.csv")
FEATURE_DATA_PATH = Path("data/processed/btcusdt_1h_features.csv")
SUMMARY_PATH = Path("artifacts/metrics/backtest_summary.csv")
EQUITY_PATH = Path("artifacts/metrics/backtest_equity_curve.csv")
TRADES_PATH = Path("artifacts/metrics/backtest_trades.csv")

# Illustrative assumptions only; change these later as a research experiment.
MODEL_NAME = "random_forest"
CONFIDENCE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
TRANSACTION_COST_SCENARIOS = [
    (0.001, 0.0005), # 0.10% fee + 0.05% slippage per side
    (0.001, 0.001),  # 0.10% fee + 0.10% slippage per side
    (0.002, 0.002),  # 0.20% fee + 0.20% slippage per side
]
INITIAL_CAPITAL = 10_000.0


def max_drawdown(equity: pd.Series) -> float:
    """Return the largest peak-to-trough percentage decline (negative value)."""
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1
    return float(drawdown.min())


def main() -> None:
    if not PROBABILITIES_PATH.exists() or not FEATURE_DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing prediction or feature data. Run feature generation and "
            "walk_forward.py first."
        )

    predictions = pd.read_csv(PROBABILITIES_PATH, parse_dates=["timestamp"])
    predictions = predictions[predictions["model"] == MODEL_NAME].copy()
    if predictions.empty:
        raise ValueError(f"No out-of-fold predictions found for {MODEL_NAME!r}.")

    outcomes = pd.read_csv(
        FEATURE_DATA_PATH,
        usecols=["timestamp", "close", "next_return", "target_up"],
        parse_dates=["timestamp"],
    )
    data = predictions.merge(
        outcomes,
        on="timestamp",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", "_outcome"),
    ).sort_values("timestamp").reset_index(drop=True)
    data = data.dropna(subset=["next_return", "target_up"]).copy()

    elapsed_years = max(
        (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]).total_seconds()
        / (365.25 * 24 * 60 * 60),
        1 / (365.25 * 24),
    )

    summaries = []
    trade_frames = []
    equity_frames = []
    for fee_per_side, slippage_per_side in TRANSACTION_COST_SCENARIOS:
        cost_per_side = fee_per_side + slippage_per_side
        cost_scenario = f"fee_{fee_per_side:.4f}_slippage_{slippage_per_side:.4f}"

        # Same-period buy-and-hold benchmark; include entry and final exit costs.
        buy_hold_returns = data["next_return"].copy()
        buy_hold_returns.iloc[0] = (1 + buy_hold_returns.iloc[0]) * (1 - cost_per_side) - 1
        buy_hold_returns.iloc[-1] = (1 + buy_hold_returns.iloc[-1]) * (1 - cost_per_side) - 1
        buy_hold_equity = INITIAL_CAPITAL * (1 + buy_hold_returns).cumprod()
        buy_hold_cagr = (buy_hold_equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / elapsed_years) - 1
        buy_hold_max_dd = max_drawdown(buy_hold_equity)

        for threshold in CONFIDENCE_THRESHOLDS:
            run = data.copy()
            # Long if predicted probability of an up hour meets threshold; otherwise cash.
            run["position"] = (run["probability_up"] >= threshold).astype(int)
            run["previous_position"] = run["position"].shift(1, fill_value=0)
            run["position_change"] = (run["position"] - run["previous_position"]).abs()

            gross_return = run["position"] * run["next_return"]
            run["strategy_return"] = (
                (1 + gross_return) * (1 - cost_per_side) ** run["position_change"] - 1
            )
            final_liquidation = int(run["position"].iloc[-1] == 1)
            if final_liquidation:
                run.loc[run.index[-1], "strategy_return"] = (
                    (1 + run["strategy_return"].iloc[-1]) * (1 - cost_per_side) - 1
                )

            run["strategy_equity"] = INITIAL_CAPITAL * (1 + run["strategy_return"]).cumprod()
            run["buy_hold_equity"] = buy_hold_equity
            run["threshold"] = threshold
            run["cost_scenario"] = cost_scenario

            strategy_cagr = (
                (run["strategy_equity"].iloc[-1] / INITIAL_CAPITAL) ** (1 / elapsed_years) - 1
            )
            summaries.append(
                {
                    "model": MODEL_NAME,
                    "confidence_threshold": threshold,
                    "fee_per_side_pct": 100 * fee_per_side,
                    "slippage_per_side_pct": 100 * slippage_per_side,
                    "validation_rows": len(run),
                    "time_in_market_pct": 100 * run["position"].mean(),
                    "position_changes_including_final_liquidation": int(
                        run["position_change"].sum() + final_liquidation
                    ),
                    "ending_strategy_equity": run["strategy_equity"].iloc[-1],
                    "strategy_total_return_pct": 100 * (
                        run["strategy_equity"].iloc[-1] / INITIAL_CAPITAL - 1
                    ),
                    "strategy_cagr_pct": 100 * strategy_cagr,
                    "strategy_max_drawdown_pct": 100 * max_drawdown(run["strategy_equity"]),
                    "buy_hold_ending_equity": buy_hold_equity.iloc[-1],
                    "buy_hold_total_return_pct": 100 * (
                        buy_hold_equity.iloc[-1] / INITIAL_CAPITAL - 1
                    ),
                    "buy_hold_cagr_pct": 100 * buy_hold_cagr,
                    "buy_hold_max_drawdown_pct": 100 * buy_hold_max_dd,
                }
            )

            trades = run[run["position_change"] == 1].copy()
            trades["action"] = trades["position"].map({1: "enter_long", 0: "exit_to_cash"})
            if final_liquidation:
                liquidation = run.iloc[[-1]].copy()
                liquidation["action"] = "liquidate_at_end"
                trades = pd.concat([trades, liquidation], ignore_index=True)
            trades["threshold"] = threshold
            trades["cost_scenario"] = cost_scenario
            trade_frames.append(
                trades[[
                    "threshold", "cost_scenario", "timestamp", "action",
                    "probability_up", "next_return", "strategy_return",
                ]]
            )
            equity_frames.append(
                run[[
                    "threshold", "cost_scenario", "timestamp", "probability_up",
                    "position", "next_return", "strategy_return", "strategy_equity",
                    "buy_hold_equity",
                ]]
            )

    summary = pd.DataFrame(summaries)
    trades_output = pd.concat(trade_frames, ignore_index=True)
    equity_output = pd.concat(equity_frames, ignore_index=True)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    equity_output.to_csv(EQUITY_PATH, index=False)
    trades_output.to_csv(TRADES_PATH, index=False)

    print(f"Model: {MODEL_NAME}; thresholds compared: {', '.join(f'{x:.0%}' for x in CONFIDENCE_THRESHOLDS)}")
    print(f"Validation period rows per scenario: {len(data):,}; final test remains untouched")
    print("Cost scenarios (fee + slippage per side): 0.10%+0.05%, 0.10%+0.10%, 0.20%+0.20%")
    print("\nHypothetical validation comparisons (benchmark pays matching entry/exit costs):")
    print(summary[[
        "fee_per_side_pct", "slippage_per_side_pct", "confidence_threshold", "time_in_market_pct",
        "position_changes_including_final_liquidation", "ending_strategy_equity",
        "strategy_total_return_pct", "strategy_max_drawdown_pct",
        "buy_hold_ending_equity", "buy_hold_max_drawdown_pct",
    ]].round(3).to_string(index=False))
    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved equity curve to: {EQUITY_PATH}")
    print(f"Saved position changes to: {TRADES_PATH}")


if __name__ == "__main__":
    main()
