
# backtest. Start with horizon 1.
# assuming entry exactly at close_t may be too optimistic. Next available price plus slippage.
# Convert forecasts into a trading rule.

# Use autogluon to fine-tune the model to our market data.
# Keep the  zero-shot model for comparison.

import pandas as pd
import numpy as np
"""
Computes the returns for a sequence of candlestick closes.
@param close An array of closing values for candlesticks.

@Note Returns are the relative change in closing value between observation k and k + 1.
@Note Assumes close is ordered by most recent observation to least recent observation.
"""
def get_returns(close: pd.Series) -> pd.Series:
    return pd.Series(-np.diff(close) / close.iloc[1:]).reset_index(drop=True)

"""
Computes relative strength index w/ wilders smoothing for an array of price history.

@param prices An array of price history.
@param n The number of candlesticks to use in the computations.
"""
def smoothed_rsi(prices: pd.Series, period: int = 14) -> np.ndarray:
    rsi = np.full(len(prices), np.nan)

    if (period <= 0):
        raise ValueError("Period must be greater than zero.")

    if (len(prices) <= period): return rsi

    diff = -np.diff(prices)

    avg_gain = np.full(len(prices), np.nan)
    avg_loss = np.full(len(prices), np.nan)

    k: int = len(prices) - period - 1
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

    return rsi


"""
Computes the p period simple moving average

@param prices An array of price history.
@param p Period length.

@note 
    SMA_M: M \\to M-p+1 = 1/p \\sum_{i=m-p+1}^M x_i
    SMA_{M+1}: M+1 \\to m-p+2 = SMA_{M} + 1/p(x_{M+1} - x_{M-p+1})
"""
def sma(prices: pd.Series, p: int = 20) -> np.ndarray:
    if p <= 0 or p > len(prices):
        return np.array([])

    n: int = len(prices)
    sma: np.ndarray = np.full(n, np.nan)

    k: int = n - p
    sma[k] = np.sum(prices.iloc[k : k + p]) / p

    for i in range(k, 0, -1):
        sma[i - 1] = sma[i] + 1 / p * (prices.iloc[i - 1] - prices.iloc[i - 1 + p])

    return sma

"""
Computes the p period weighted moving average.
The computation is simply a weighted mean

@param prices An array of price history.
@param p Period length.
"""
def wma(prices: pd.Series, period: int = 14) -> np.ndarray:
    if (period <= 0):
        raise ValueError("Period must be greater than zero.")

    W = np.full(len(prices), np.nan)

    if (len(prices) < period):
        return W
    
    weights = np.arange(period, 0, -1)
    weight_sum = weights.sum()

    for i in range(len(prices) - period + 1):
        window = prices.iloc[i: i + period]
        W[i] = np.sum(window * weights)  / weight_sum
    return W


"""
Computes the p period exponential moving average.

@param prices An array of price history.
@param p Period length.

@note EMA_t = {
    p_1 if t = 1,
    \\alpha p_t + (1-\\alpha)EMA_{t-1} o.w
}

More info can be found in the repos reference doc.
"""
def ema(prices: pd.Series, period: int = 15) -> pd.Series:
    if (len(prices) == 0): return pd.Series(index=prices.index)

    n: int = len(prices)-1
    alpha: np.float128 = np.float128(2) / np.float128((period + 1))
    ret: np.ndarray = np.full(n+1, np.nan)

    ret[n] = prices.iloc[n]
    for i in range(n-1, -1, -1): 
        ret[i] = alpha * prices.iloc[i] + (1-alpha) * ret[i+1]

    return pd.Series(ret, index=prices.index)

"""
Calcuates Bollinger Band Percent B (%B).

@param prices Price observations ordered from newest to oldest.
@param period Number of observations used to calculate the moving average and standard deviation.
@param std Number of standard deviations between the moving averages and each Bollinger Band.

@return Array of %b values.
"""
def percent_b(prices: pd.Series, period: int = 20, std: int = 2) -> pd.Series:
    if (period < 0 or period > len(prices)):
        return pd.Series([])

    rolling_mean: np.ndarray = np.convolve(
        prices, np.ones(period) / period, mode="valid"
    )

    rolling_std: np.ndarray = np.std(
        [prices.iloc[i : i + period] for i in range(len(prices) - period + 1)],
        axis=1,
    )

    upper_band = rolling_mean + std * rolling_std
    lower_band = rolling_mean - std * rolling_std
    ret =  pd.Series((prices.iloc[:len(prices) - period + 1] - lower_band) / (upper_band - lower_band))
    return ret

def msi(): 
    ...
"""
Calculates VWAP.

@param

@note 

"""
def vwap(df: pd.DataFrame) -> pd.Series: 
    data = df.copy()
    data['price'] = (df['high'] + df['low'] + df['close']) / 3

    def calcuate_vwap(group: pd.DataFrame) -> pd.DataFrame:
        vwap = (group['price'] * group['volume']).cumsum() / group['volume'].cumsum()
        group['VWAP'] = vwap
        return group

    data = data.sort_values("real_timestamp")

    vwap_df: pd.DataFrame = data.groupby('session_end_date').apply(calcuate_vwap, include_groups=False) # type:ignore

    vwap_df = vwap_df.sort_values('real_timestamp', ascending=False)
    return vwap_df['VWAP'].reset_index(drop=True)


"""
Computes a rolling window standard deviation as a measure of market volatility on recent movements.
"""
def rolling_std(returns: pd.Series, period: int=14) -> pd.Series:
    if len(returns) < period:
        raise ValueError(f"Period cannot exceed length of the input Series. {period} > {len(returns)}") 
    # Log Returns? 
    rolling_std = np.std(
        [returns.iloc[i: i + period] for i in range(0, len(returns) - period + 1)],
        axis=1,
    )

    result = pd.Series(np.nan, index=returns.index, dtype=float)
    result.iloc[:len(rolling_std)] = rolling_std
    return result

def alpha():
    ...
def beta():
    ...
