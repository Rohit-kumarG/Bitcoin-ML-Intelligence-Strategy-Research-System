"""Long/short 1x perpetual simulation using futures OOF probabilities and funding."""

from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path("artifacts/metrics/futures_walk_forward_probabilities.csv")
FEATURES_PATH = Path("data/processed/btcusdt_usdm_1h_features.csv")
FUNDING_PATH = Path("data/raw/btcusdt_usdm_funding.csv")
SUMMARY_PATH = Path("artifacts/metrics/futures_long_short_summary.csv")
EQUITY_PATH = Path("artifacts/metrics/futures_long_short_equity.csv")
TRADES_PATH = Path("artifacts/metrics/futures_long_short_trades.csv")

THRESHOLDS = [0.55, 0.60, 0.65, 0.70]
COST_SCENARIOS = [(0.001, 0.0005), (0.001, 0.001), (0.002, 0.002)]
INITIAL_CAPITAL = 10_000.0
EXPOSURE_MULTIPLE = 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def main() -> None:
    for path in [PREDICTIONS_PATH, FEATURES_PATH, FUNDING_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run futures download, feature, and walk-forward scripts first."
            )

    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"])
    outcomes = pd.read_csv(
        FEATURES_PATH,
        usecols=["timestamp", "next_return", "target_up"],
        parse_dates=["timestamp"],
    )
    data = predictions.merge(
        outcomes, on="timestamp", how="left", validate="one_to_one"
    ).dropna(subset=["next_return", "target_up"]).sort_values("timestamp").reset_index(drop=True)

    funding = pd.read_csv(FUNDING_PATH, parse_dates=["timestamp"])
    funding["timestamp"] = funding["timestamp"] - pd.Timedelta(hours=2)
    funding = funding.rename(columns={"funding_rate": "settlement_funding_rate"})
    data = data.merge(
        funding[["timestamp", "settlement_funding_rate"]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    data["settlement_funding_rate"] = data["settlement_funding_rate"].fillna(0.0)

    elapsed_years = max(
        (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]).total_seconds()
        / (365.25 * 24 * 60 * 60),
        1 / (365.25 * 24),
    )
    buy_hold = data["next_return"].copy()
    buy_hold_endings = {}
    summaries = []
    equity_frames = []
    trade_frames = []

    for fee_per_side, slippage_per_side in COST_SCENARIOS:
        cost_per_side = fee_per_side + slippage_per_side
        cost_label = f"fee_{fee_per_side:.4f}_slippage_{slippage_per_side:.4f}"
        benchmark_returns = buy_hold.copy()
        benchmark_returns.iloc[0] = (1 + benchmark_returns.iloc[0]) * (1 - cost_per_side) - 1
        benchmark_returns.iloc[-1] = (1 + benchmark_returns.iloc[-1]) * (1 - cost_per_side) - 1
        benchmark_equity = INITIAL_CAPITAL * (1 + benchmark_returns).cumprod()
        buy_hold_endings[cost_label] = benchmark_equity.iloc[-1]

        for threshold in THRESHOLDS:
            run = data.copy()
            run["position"] = 0
            run.loc[run["probability_up"] >= threshold, "position"] = 1
            run.loc[run["probability_up"] <= 1 - threshold, "position"] = -1
            run["previous_position"] = run["position"].shift(1, fill_value=0)
            run["position_change_units"] = (
                run["position"] - run["previous_position"]
            ).abs()

            # Funding timestamp is aligned to the one-hour holding interval ending
            # at settlement; current exposure pays positive funding and receives negative.
            run["price_pnl_return"] = EXPOSURE_MULTIPLE * run["position"] * run["next_return"]
            run["funding_pnl_return"] = (
                -EXPOSURE_MULTIPLE
                * run["position"]
                * run["settlement_funding_rate"]
            )
            run["net_return"] = (
                (1 + run["price_pnl_return"] + run["funding_pnl_return"])
                * (1 - cost_per_side) ** run["position_change_units"]
                - 1
            )

            final_position = int(run["position"].iloc[-1])
            final_liquidation_units = abs(final_position)
            if final_liquidation_units:
                run.loc[run.index[-1], "net_return"] = (
                    (1 + run["net_return"].iloc[-1])
                    * (1 - cost_per_side) ** final_liquidation_units
                    - 1
                )

            run["equity"] = INITIAL_CAPITAL * (1 + run["net_return"]).cumprod()
            run["buy_hold_equity"] = benchmark_equity
            run["threshold"] = threshold
            run["cost_scenario"] = cost_label

            total_return = run["equity"].iloc[-1] / INITIAL_CAPITAL - 1
            cagr = (run["equity"].iloc[-1] / INITIAL_CAPITAL) ** (1 / elapsed_years) - 1
            active = run["position"] != 0
            summaries.append(
                {
                    "threshold": threshold,
                    "fee_per_side_pct": 100 * fee_per_side,
                    "slippage_per_side_pct": 100 * slippage_per_side,
                    "validation_rows": len(run),
                    "long_hours": int((run["position"] == 1).sum()),
                    "short_hours": int((run["position"] == -1).sum()),
                    "time_in_market_pct": 100 * active.mean(),
                    "position_change_units_including_final_exit": float(
                        run["position_change_units"].sum() + final_liquidation_units
                    ),
                    "funding_settlements_held": int(
                        ((run["settlement_funding_rate"] != 0) & active).sum()
                    ),
                    "ending_equity": run["equity"].iloc[-1],
                    "total_return_pct": 100 * total_return,
                    "cagr_pct": 100 * cagr,
                    "max_drawdown_pct": 100 * max_drawdown(run["equity"]),
                    "buy_hold_ending_equity": benchmark_equity.iloc[-1],
                    "buy_hold_max_drawdown_pct": 100 * max_drawdown(benchmark_equity),
                    "funding_cashflow_pct_not_compounded": 100
                    * run["funding_pnl_return"].sum(),
                }
            )

            changes = run[run["position_change_units"] > 0].copy()
            changes["action"] = changes.apply(
                lambda row: "enter_long" if row["position"] == 1 and row["previous_position"] == 0
                else "enter_short" if row["position"] == -1 and row["previous_position"] == 0
                else "exit_to_cash" if row["position"] == 0
                else "reverse_position",
                axis=1,
            )
            if final_liquidation_units:
                final_row = run.iloc[[-1]].copy()
                final_row["action"] = "liquidate_at_end"
                changes = pd.concat([changes, final_row], ignore_index=True)
            changes["threshold"] = threshold
            changes["cost_scenario"] = cost_label
            trade_frames.append(
                changes[[
                    "threshold", "cost_scenario", "timestamp", "action", "position",
                    "probability_up", "next_return", "settlement_funding_rate", "net_return",
                ]]
            )
            equity_frames.append(
                run[[
                    "threshold", "cost_scenario", "timestamp", "probability_up", "position",
                    "next_return", "settlement_funding_rate", "price_pnl_return",
                    "funding_pnl_return", "net_return", "equity", "buy_hold_equity",
                ]]
            )

    summary = pd.DataFrame(summaries)
    equity_output = pd.concat(equity_frames, ignore_index=True)
    trades_output = pd.concat(trade_frames, ignore_index=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    equity_output.to_csv(EQUITY_PATH, index=False)
    trades_output.to_csv(TRADES_PATH, index=False)

    print("Strategy: futures-trained RF, long/short/flat, 1x notional exposure")
    print(f"OOF validation rows: {len(data):,}; final test remains reserved")
    print("Positive funding: longs pay, shorts receive; historical funding settlements aligned to holding intervals")
    print("Fee/slippage scenarios per side: 0.10%+0.05%, 0.10%+0.10%, 0.20%+0.20%")
    print("\nValidation results:")
    print(summary[[
        "threshold", "fee_per_side_pct", "slippage_per_side_pct", "long_hours",
        "short_hours", "time_in_market_pct", "position_change_units_including_final_exit",
        "ending_equity", "total_return_pct", "max_drawdown_pct",
        "buy_hold_ending_equity", "funding_cashflow_pct_not_compounded",
    ]].round(3).to_string(index=False))
    print(f"\nSaved summary to: {SUMMARY_PATH}")
    print(f"Saved equity rows to: {EQUITY_PATH}")
    print(f"Saved position changes to: {TRADES_PATH}")


if __name__ == "__main__":
    main()
