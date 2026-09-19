import pandas as pd
import numpy as np
"""
Computes a rolling window standard deviation as a measure of market volatility on recent movements.
"""
def rolling_std(returns: pd.Series, period: int=14) -> pd.Series:
    if (period <= 0 or period > len(returns)):
        raise ValueError("period must be between 1 and len(prices).")

    # Log Returns? 
    output = np.full(len(returns), np.nan)
    for positions in returns.groupby(level=0).indices.values():
        current = returns.iloc[np.array(positions)]

        if (len(current) < period):
            continue

        count = len(current) - period + 1

        rolling_std = np.std(
                [current.iloc[i: i + period] for i in range(0, count)],
                axis=1,
        )

        output[positions[:count]] = rolling_std

    return pd.Series(output, index=returns.index, name="rolling_std")


"""
Calcuates Bollinger Band Percent B (%B).

@param prices Price observations ordered from newest to oldest.
@param period Number of observations used to calculate the moving average and standard deviation.
@param std Number of standard deviations between the moving averages and each Bollinger Band.

@return Array of %b values.
"""
def percent_b(prices: pd.Series, period: int = 20, std: int = 2) -> pd.Series:
    if (period <= 0 or period > len(prices)):
        raise ValueError("period must be between 1 and len(prices).")

    output: np.ndarray = np.full(len(prices), np.nan)
    weights = np.ones(period) / period

    for positions in prices.groupby(level=0).indices.values():
        price = prices.iloc[np.array(positions)]

        if len(price) < period:
            continue

        count: int = len(price) - period + 1

        rolling_mean: np.ndarray = np.convolve(
            price, weights, mode="valid"
        )

        rolling_std: np.ndarray = np.std(
            [price.iloc[i : i + period] for i in range(count)],
            axis=1,
        )

        lower_band: np.ndarray = rolling_mean - std * rolling_std
        band_width = 2 * std * rolling_std

        values = np.full(count, np.nan)
        np.divide(
                (price.iloc[:count] - lower_band),
                band_width,
                out=values,
                where=band_width > 0
        )

        output[positions[:count]] = values

    return pd.Series(output, index=prices.index, name="percent_b")
