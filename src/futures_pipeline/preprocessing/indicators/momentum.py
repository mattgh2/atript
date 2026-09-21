import pandas as pd
import numpy as np
from .trend import sma

"""
Computes relative strength index w/ wilders smoothing for price history.

@param prices An array of price history.
@param n The number of candlesticks to use in the computations.
@Note: Input should be passed in descending order by window_start.
@Note: Prices indices must be symbol labels.
"""
def smoothed_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    if (period <= 0):
        raise ValueError("Period must be greater than zero.")
    if not all(isinstance(label, str) for label in prices.index):
        raise ValueError("The index of the input series of prices must be futures symbols.")

    
    output = np.full(len(prices), np.nan)

    for positions in prices.groupby(level=0, sort=False).indices.values():
        price = prices.iloc[np.array(positions)]
        rsi = np.full(len(price), np.nan)

        if (len(price) <= period): 
            continue

        diff = -np.diff(price)

        avg_gain = np.full(len(price), np.nan)
        avg_loss = np.full(len(price), np.nan)

        k: int = len(price) - period - 1
        window = diff[k: k + period]
        avg_gain[k] = np.sum(window[window > 0]) / period
        avg_loss[k] = -np.sum(window[window < 0]) / period

        if avg_gain[k] == 0 and avg_loss[k] == 0:
            rsi[k] = 50.0
        elif avg_loss[k] == 0:
            rsi[k] = 100.0
        else:
            RS = avg_gain[k] / avg_loss[k]
            rsi[k] = 100 * (RS / (1 + RS))

        for i in range(k, 0, -1):
            gain = max(diff[i - 1], 0)
            loss =  max(-diff[i - 1], 0)

            avg_gain[i - 1] = ((period - 1) * avg_gain[i] + gain) / period
            avg_loss[i-1] = ((period - 1) * avg_loss[i] + loss) / period

            if avg_gain[i - 1] == 0 and avg_loss[i - 1] == 0:
                rsi[i - 1] = 50.0
            elif avg_loss[i - 1] == 0:
                rsi[i - 1] = 100.0
            else:
                RS = avg_gain[i - 1] / avg_loss[i - 1]
                rsi[i - 1] = 100 * (RS / (1 + RS))

        output[positions] = rsi

    return pd.Series(output, prices.index, name="rsi")



# Fast stochastic oscillator (Fast %K).
def fast_k(data: pd.DataFrame, period: int = 14):
    output = np.full(data.shape[0], np.nan)
    for positions in data.groupby('ticker').indices.values():
        df = data.iloc[positions]
        for i in range(df.shape[0] - period + 1):
            low = df.iloc[i : i + period]['low'].min()
            high = df.iloc[i : i + period]['high'].max()
            percent_k = (df.iloc[i]['close'] - low) / (high - low) * 100
            output[positions[i]] = percent_k
    return pd.Series(output, index=data.index, name="fast_k")


# Slow stochastic oscillator (Slow %K).
def slow_k(fast_k: pd.Series):
    output = np.full(len(fast_k), np.nan)
    for positions in fast_k.groupby(level=0).indices.values():
        current = fast_k.iloc[np.array(positions)]
        slow = current.iloc[::-1].rolling(3, min_periods=3).mean().iloc[::-1]
        output[positions] = slow
    return pd.Series(output, index=fast_k.index, name='slow_k')


# Rate of Change
def RoC(prices: pd.Series, period: int = 14):
    output = np.full(len(prices), np.nan)
    for positions in prices.groupby(level=0).indices.values():
        price = prices.iloc[np.array(positions)]
        if len(price) < period:
            continue
        roc = np.full(len(price), np.nan)
        for i in range(len(price) - period):
            roc[i] = 100 * (price.iloc[i] / price.iloc[i + period] - 1)
        output[positions] = roc

    return pd.Series(output, index=prices.index, name="RoC")

# Commodity Channel Index
def cci(data: pd.DataFrame, period: int = 20):
    output = np.full(data.shape[0], np.nan)
    for positions in data.groupby('ticker').indices.values():
        current = data.iloc[np.array(positions)].copy()
        current["TP"] = (current['high']  + current['low'] + current['close']) / 3
        current['sma'] = sma(current['TP'], p=period)
        current['mad'] = current['TP'].iloc[::-1].rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).iloc[::-1].to_numpy()
        output[positions] = (current['TP'] - current['sma']) / (0.015 * current['mad'])
    return pd.Series(output, index=data.index, name="cci")


