from ..typedefs import PredictionInterval, ContractSpec, BacktestResults, TradingFees, Signal
from ..config import load_fees
import pandas as pd
import numpy as np
"""
Performs Backtest model evaluation on a set of forecasts.

Metrics:
    - Trade Count
    - Net pnl
    - Win rate
    - Average pnl
    - Gross profits
    - Gross loss

@param
@param
@param
@param
@param
"""
def backtest_strategy(
    forecasts: pd.DataFrame,
    market_data: pd.DataFrame,
    pred_interval: PredictionInterval,
    contract_spec: ContractSpec,
    pred_length: int
) -> tuple[pd.DataFrame, BacktestResults]:
    fees: TradingFees = load_fees()

    round_trip_cost: float = (
        fees.entry_fee
        + fees.exit_fee
        + fees.entry_commission
        + fees.exit_commission
    )

    execution_cost = (
            fees.spread + 2 * fees.slippage
    ) * contract_spec.price_per_tick

    total_round_trip_cost: float = round_trip_cost + execution_cost

    forecasts = forecasts.sort_values(["ticker", "forecast_origin"]).reset_index(
        drop=True
    )
    market_data = market_data.sort_values(['ticker', 'model_timestamp']).reset_index(drop=True)

    market = market_data[['ticker', 'model_timestamp', 'open', 'close']]

    entry = market[['ticker', 'model_timestamp']].copy()

    entry['entry_price'] = market.groupby('ticker')['open'].shift(-1)
    entry = entry.rename(columns={'model_timestamp': 'forecast_origin'})

    exit = market[["ticker", "model_timestamp", "close"]].rename(
        columns={"close": "exit_price"}
    )

    forecasts = forecasts.merge(
            entry,
            on=['ticker', 'forecast_origin'],
            how='left',
            validate="many_to_one"
    )

    forecasts = forecasts.merge(
            exit,
            on=['ticker', 'model_timestamp'],
            how='left',
            validate="many_to_one"
    )

    #  Matched origin whose next origin is missing leaves entry_price as NaN.
    # All forecasts should not have NaN's. It is possible that the entry price is missing.

    signal_df = forecasts[forecasts["horizon"].eq(pred_length)].reset_index(drop=True)

    signal_df["signal"] = pd.Series(
        # calculate_signal_point(row['threshold'], row['predictions'] - row['origin_close'])
        calcuate_signal_pi(
            row["origin_close"],
            total_round_trip_cost,
            contract_spec,
            row[str(pred_interval.lower)],
            row[str(pred_interval.upper)],
        )
        for _, row in signal_df.iterrows()
    )

    # For Sanity.
    # for idx, row in signal_df.iterrows():
    #     print(
    #         f"Threshold: {row['threshold']:.8f}. Predicted close: {row['predictions']:.8f}. "
    #         f"Actual close: {row['exit_price']:.8f}. AE: {abs(row['exit_price'] - row['predictions'])}. "
    #         f"Signal is {row['signal']}"
    #     )

    missing_fill = signal_df['entry_price'].isna() | signal_df['exit_price'].isna()

    signal_df = signal_df.loc[~missing_fill].reset_index(drop=True)

    signal_df["gross_pnl"] = (
        signal_df["signal"]
        * (signal_df["exit_price"] - signal_df["entry_price"])
        * contract_spec.multiplier
    )

    signal_df['trading_cost'] = np.where(
            signal_df['signal'] != Signal.HOLD,
            total_round_trip_cost,
            0.0
    )

    signal_df['net_pnl'] = signal_df['gross_pnl'] - signal_df['trading_cost']


    allocated_capital: float = 1000.0
    risk_free_per_window: float = 0.0
    returns: pd.Series = signal_df['net_pnl'] / allocated_capital
    excess: pd.Series = returns - risk_free_per_window
    std = excess.std(ddof=1)
    net_sharpe_per_window = excess.mean() / std if len(excess) >= 2 and std > 0 else float('nan')

    trades = signal_df[signal_df['signal'] != Signal.HOLD]
    trade_count = trades.shape[0]
    net_pnl = trades['net_pnl'].sum()
    win_rate = (trades['net_pnl'] > 0).mean()
    average_pnl = trades['net_pnl'].mean()

    gross_profit = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = -trades[trades['net_pnl'] < 0]['net_pnl'].sum()

    return (
        signal_df,
        BacktestResults(
            trade_count,
            net_pnl,
            win_rate,
            average_pnl,
            gross_profit,
            gross_loss,
            missing_fill.sum(),
            net_sharpe_per_window
        ),
    )

def calcuate_signal_pi(
    origin_price, 
    total_round_trip_cost: float, 
    contract_spec: ContractSpec,
    lower_b: float,
    upper_b: float,
    buffer: float = 0
) -> Signal:
    required_move_points = total_round_trip_cost / contract_spec.multiplier + buffer
    if lower_b - origin_price > required_move_points:
        signal = Signal.LONG
    elif origin_price - upper_b > required_move_points:
        signal = Signal.SHORT
    else:
        signal = Signal.HOLD
    return signal

# def calculate_signal_point(threshold, diff):
#     if diff > threshold:
#         signal = Signal.LONG
#     elif diff < -threshold:
#         signal = Signal.SHORT
#     else:
#         signal = Signal.HOLD
#     return signal
