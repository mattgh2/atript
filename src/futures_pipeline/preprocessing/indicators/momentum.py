import pandas as pd
import numpy as np

"""
Computes relative strength index w/ wilders smoothing for price history.

@param prices An array of price history.
@param n The number of candlesticks to use in the computations.
@Note: Input should be passed in descending order by window_start.
@Note: Prices indices must be symbol labels.
"""

"""
Calculate rsi for contiguous groups of prices. 
If there is a nan value in the period, then the rsi should be nan.
"""
def smoothed_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    if (period <= 0):
        raise ValueError("Period must be greater than zero.")
    if not all(isinstance(label, str) for label in prices.index):
        raise ValueError("The index of the input series of prices must be futures symbols.")
    def compute_rsi(avg_gain, avg_loss) -> float:
        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0

        RS = avg_gain / avg_loss
        return 100 * (RS / (1 + RS))
    
    output = np.full(len(prices), np.nan)

    for positions in prices.groupby(level=0, sort=False).indices.values():
        price = prices.iloc[np.array(positions)]
        rsi = np.full(len(price), np.nan)

        if (len(price) <= period): 
            continue

        diff = -np.diff(price)

        avg_gain = np.full(len(price), np.nan)
        avg_loss = np.full(len(price), np.nan)
        
        k = len(price) - period - 1

        window = diff[k: k + period]

        if not np.any(np.isnan(window)):
            avg_gain[k] = np.sum(window[window > 0]) / period
            avg_loss[k] = -np.sum(window[window < 0]) / period
            rsi[k] = compute_rsi(avg_gain[k], avg_loss[k])

        for i in range(k - 1, -1, -1):

            ''' avg_gain[i + 1], avg_loss[i + 1] may be nan if there was a missing row in its period. 
                Must compute the average gain and loss using the standard approach.'''

            # If avg_gain[i + 1] is nan, then avg_loss[i + 1] is guaranteed nan. 
            # Do not need explicit if.
            if np.isnan(avg_gain[i + 1]):
                window = diff[i: i + period]
                if np.any(np.isnan(window)):
                    continue
                avg_gain[i] = np.sum(window[window > 0]) / period
                avg_loss[i] = -np.sum(window[window < 0]) / period
            else:
                gain = max(diff[i], 0)
                loss = max(-diff[i], 0)
                avg_gain[i] = ((period - 1) * avg_gain[i + 1] + gain) / period
                avg_loss[i] = ((period - 1) * avg_loss[i + 1] + loss) / period

            rsi[i] = compute_rsi(avg_gain[i], avg_loss[i])

        output[positions] = rsi

    return pd.Series(output, prices.index, name="rsi")


# Fast stochastic oscillator (Fast %K).
def fast_k(data: pd.DataFrame, period: int = 14):
    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    output = np.full(data.shape[0], np.nan)
    for positions in data.groupby('ticker').indices.values():
        df = data.iloc[positions]
        for i in range(df.shape[0] - period + 1):
            window = df.iloc[i : i + period]
            if window[['high', 'low', 'close']].isna().to_numpy().any():
                continue

            low = window['low'].min()
            high = window['high'].max()
            if high == low:
                continue

            percent_k = (window.iloc[0]['close'] - low) / (high - low) * 100
            output[positions[i]] = percent_k
    return pd.Series(output, index=data.index, name="fast_k")


# Slow stochastic oscillator (Slow %K).
def slow_k(fast_k: pd.Series):
    output = np.full(len(fast_k), np.nan)
    for positions in fast_k.groupby(level=0).indices.values():
        current = fast_k.iloc[np.array(positions)]
        for i in range(len(current) - 2):
            window = current.iloc[i : i + 3]
            if window.isna().any():
                continue
            output[positions[i]] = window.mean()
    return pd.Series(output, index=fast_k.index, name='slow_k')


# Rate of Change
def RoC(prices: pd.Series, period: int = 14):
    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    output = np.full(len(prices), np.nan)
    for positions in prices.groupby(level=0).indices.values():
        price = prices.iloc[np.array(positions)]
        if len(price) <= period:
            continue
        roc = np.full(len(price), np.nan)
        for i in range(len(price) - period):
            window = price.iloc[i : i + period + 1]
            if window.isna().any() or window.iloc[-1] == 0:
                continue
            roc[i] = 100 * (window.iloc[0] / window.iloc[-1] - 1)
        output[positions] = roc

    return pd.Series(output, index=prices.index, name="RoC")

# Commodity Channel Index
def cci(data: pd.DataFrame, period: int = 20):
    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    output = np.full(data.shape[0], np.nan)
    for positions in data.groupby('ticker').indices.values():
        current = data.iloc[np.array(positions)]
        for i in range(len(current) - period + 1):
            window = current.iloc[i : i + period][['high', 'low', 'close']]
            if window.isna().to_numpy().any():
                continue

            typical_prices = window.mean(axis=1)
            average_price = typical_prices.mean()
            mean_deviation = (typical_prices - average_price).abs().mean()
            if mean_deviation == 0:
                continue

            output[positions[i]] = (
                (typical_prices.iloc[0] - average_price)
                / (0.015 * mean_deviation)
            )
    return pd.Series(output, index=data.index, name="cci")


