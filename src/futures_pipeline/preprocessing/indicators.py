
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

"""
Computes the p period simple moving average

@param prices An array of price history.
@param p Period length.

@note 
    SMA_M: M \\to M-p+1 = 1/p \\sum_{i=m-p+1}^M x_i
    SMA_{M+1}: M+1 \\to m-p+2 = SMA_{M} + 1/p(x_{M+1} - x_{M-p+1})
"""
def sma(prices: pd.Series, p: int = 20) -> pd.Series:
    if p <= 0 or p > len(prices):
        raise ValueError("p must be between 1 and len(prices)")

    output = np.full(len(prices), np.nan)
    for positions in prices.groupby(level=0).indices.values():
        price = prices.iloc[np.array(positions)]

        n: int = len(price)

        if n < p:
            continue

        sma: np.ndarray = np.full(n, np.nan)

        k: int = n - p
        sma[k] = np.sum(price.iloc[k : k + p]) / p

        for i in range(k, 0, -1):
            sma[i - 1] = sma[i] + 1 / p * (price.iloc[i - 1] - price.iloc[i - 1 + p])

        output[positions] = sma

    return pd.Series(output, index=prices.index, name="sma")

"""
Computes the p period weighted moving average.
The computation is simply a weighted mean

@param prices An array of price history.
@param p Period length.
"""
def wma(prices: pd.Series, period: int = 14) -> pd.Series:
    if (period <= 0):
        raise ValueError("Period must be greater than zero.")

    output = np.full(len(prices), np.nan)

    for positions in prices.groupby(level=0).indices.values():
        price = prices.iloc[np.array(positions)]
        W = np.full(len(price), np.nan)

        if (len(price) < period):
            continue
    
        weights = np.arange(period, 0, -1)
        weight_sum = weights.sum()

        for i in range(len(price) - period + 1):
            window = price.iloc[i: i + period]
            W[i] = np.sum(window * weights)  / weight_sum

        output[positions] = W

    return pd.Series(output, index=prices.index, name="wma")


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
    if len(prices) < period or period <= 0:
        raise ValueError("period must be between 1 and len(prices).")


    alpha: float = 2.0 / (period + 1)
    output = np.full(len(prices), np.nan)
    for positions in prices.groupby(level=0).indices.values():
        price = prices.iloc[np.array(positions)]

        if len(price) < period:
            continue

        n: int = len(price)-1
        ema = np.full(n+1, np.nan)

        ema[n] = price.iloc[n]
        for i in range(n-1, -1, -1): 
            ema[i] = alpha * price.iloc[i] + (1-alpha) * ema[i+1]

        output[positions] = ema

    return pd.Series(output, index=prices.index, name="ema")

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
    data["_position"] = np.arange(data.shape[0])

    data = data.sort_values("real_timestamp", kind='stable')

    def calcuate_vwap(group: pd.DataFrame) -> pd.DataFrame:
        vwap = (group['price'] * group['volume']).cumsum() / group['volume'].cumsum()
        group['VWAP'] = vwap
        return group

    vwap_df: pd.DataFrame = data.groupby(['session_end_date', 'ticker']).apply(calcuate_vwap, include_groups=False) # type:ignore

    vwap = vwap_df.sort_values('_position')['VWAP'].to_numpy()
    return pd.Series(vwap, index=df.index, name="VWAP")


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

def alpha():
    ...
def beta():
    ...
