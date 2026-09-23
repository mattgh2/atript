import pandas as pd
import numpy as np
from .util import smoothing_average

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



def ema(prices: pd.Series, period: int = 15):
    return smoothing_average(prices, 2 / (period + 1), period).rename("ema")

def t3ma(closes: pd.Series, alpha: float = .7, period: int = 15):
    output: np.ndarray = np.full(len(closes), np.nan)
    c1 = -alpha ** 3
    c2 = 3 * alpha ** 2 + 3 * alpha ** 3
    c3 = -6 * alpha ** 2 - 3 * alpha -  3 * alpha ** 3
    c4 = 1 + 3 * alpha + alpha ** 3 + 3 * alpha ** 2

    e: np.ndarray = np.full(7, np.nan, dtype=object)
    for positions in closes.groupby(level=0).indices.values():
        if len(positions) < period:
            continue

        e[0] = closes.iloc[np.array(positions)]
        for i in range(1, len(e)):
            e[i] = ema(e[i-1], period)
        output[positions] = c1 * e[6] + c2 * e[5] + c3 * e[4] + c4 * e[3]

    return pd.Series(output, index=closes.index, name='t3ma')


def dma(
    sma_values: pd.Series, displacement: int = 5,
) -> pd.Series:
    if displacement < 0:
        raise ValueError("displacement must be nonnegative.")
    return (
        sma_values.groupby(level=0)
        .shift(-displacement)
        .rename("dma")
    )

