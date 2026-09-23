import pandas as pd
import numpy as np
from .util import smoothing_average, ttr
from .trend import ema
from typing import NamedTuple


class BollingerBands(NamedTuple):
    upper_band: pd.Series
    lower_band: pd.Series
    percent_b: pd.Series


class KeltnerChannel(NamedTuple):
    upper_band: pd.Series
    lower_band: pd.Series   
    bollinger_squeeze: pd.Series


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
Calcuates Bollinger Bands.

@param prices Price observations ordered from newest to oldest.
@param period Number of observations used to calculate the moving average and standard deviation.
@param std Number of standard deviations between the moving averages and each Bollinger Band.

@return Array of %b values.
"""
def bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2) -> BollingerBands:
    if (period <= 0 or period > len(prices)):
        raise ValueError("period must be between 1 and len(prices).")

    percent_b: np.ndarray = np.full(len(prices), np.nan)
    upper_band: np.ndarray = np.full(len(prices), np.nan)
    lower_band: np.ndarray = np.full(len(prices), np.nan)

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

        upper_band[positions[:count]] = rolling_mean + std * rolling_std
        lower_band[positions[:count]] = rolling_mean - std * rolling_std

        band_width = 2 * std * rolling_std

        pb = np.full(count, np.nan)

        np.divide(
                (price.iloc[:count] - lower_band[positions[:count]]),
                band_width,
                out=pb,
                where=band_width > 0
        )

        percent_b[positions[:count]] = pb

    return BollingerBands(
        upper_band=pd.Series(upper_band, index=prices.index, name="upper_band"),
        lower_band=pd.Series(lower_band, index=prices.index, name="lower_band"),
        percent_b=pd.Series(percent_b, index=prices.index, name="percent_b"),
    )


# Average True Range
def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    output: np.ndarray = np.full(data.shape[0], np.nan)
    for positions in data.groupby("ticker").indices.values():
        if len(positions) < period:
            continue
        current = data.iloc[positions].copy()
        current['TTR'] = ttr(current)
        output[positions] = smoothing_average(current['TTR'], alpha=1 / period, period=period, init='sma')

    return pd.Series(output, index=data.index, name='atr')


def vr(data: pd.DataFrame):
    ttr_values: pd.Series = ttr(data).to_numpy()
    atr_values: pd.Series = atr(data).to_numpy()
    output = np.full(data.shape[0], np.nan)
    np.divide(
            ttr_values, 
            atr_values,
            out=output,
            where=atr_values > 0
    )
    return pd.Series(output, index=data.index, name="vr")


# Keltner Channels (Upper and lower bands)
def kc(
    data: pd.DataFrame, 
    ema_period: int = 20, 
    atr_period: int = 20, 
    bollinger_period: int = 20,
    bollinger_std: int = 2,
    mult: float = 2.0
) -> KeltnerChannel:

    mid = ema(data['close'], period=ema_period).to_numpy()
    a = atr(data, atr_period).to_numpy()

    upper_kc: np.ndarray = mid + mult * a
    lower_kc: np.ndarray = mid - mult * a

    upper_b, lower_b, _ = bollinger_bands(
        data["close"], period=bollinger_period, std=bollinger_std
    )

    valid = (
        np.isfinite(upper_kc)
        & np.isfinite(lower_kc)
        & np.isfinite(upper_b)
        & np.isfinite(lower_b)
    )


    bollinger_squeeze = np.full(len(upper_b), np.nan)
    bollinger_squeeze[valid] = (upper_b[valid] < upper_kc[valid]) & (lower_b[valid] > lower_kc[valid])

    return KeltnerChannel(
        upper_band=pd.Series(upper_kc, index=data.index, name="upper_kc"),
        lower_band=pd.Series(lower_kc, index=data.index, name="lower_kc"),
        bollinger_squeeze= pd.Series(bollinger_squeeze, index=data.index, name='bollinger_squeeze')
    )
