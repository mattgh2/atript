import pandas as pd
import numpy as np
from typing import Literal
"""
Computes the p period Wilder's moving average.

@param prices An array of price history.
@param p Period length.

More info can be found in the repos reference doc.
"""
def smoothing_average(
    prices: pd.Series,
    alpha: float,
    period: int = 15,
    init: Literal["first", "sma"] = "first",
) -> pd.Series:
    if init not in ("first", "sma"):
        raise ValueError("init must be 'first' or 'sma'")

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if len(prices) < period or period <= 0:
        raise ValueError("period must be between 1 and len(prices).")

    output = np.full(len(prices), np.nan)
    for positions in prices.groupby(level=0).indices.values():
        price = prices.iloc[np.array(positions)]

        if len(price) < period:
            continue

        avg = np.full(len(price), np.nan)

        # Prices are newest-to-oldest. Process each finite contiguous segment
        # independently so a missing observation does not contaminate newer data.
        segment_end = len(price) - 1
        while segment_end >= 0:
            while segment_end >= 0 and pd.isna(price.iloc[segment_end]):
                segment_end -= 1
            if segment_end < 0:
                break

            segment_start = segment_end
            while segment_start > 0 and pd.notna(price.iloc[segment_start - 1]):
                segment_start -= 1

            if init == 'first':
                seed = segment_end
                avg[seed] = price.iloc[seed]
            else:
                if segment_end - segment_start + 1 < period:
                    segment_end = segment_start - 1
                    continue
                seed = segment_end - period + 1
                avg[seed] = price.iloc[seed : segment_end + 1].mean()

            for i in range(seed - 1, segment_start - 1, -1):
                avg[i] = alpha * price.iloc[i] + (1 - alpha) * avg[i + 1]

            segment_end = segment_start - 1

        output[positions] = avg

    return pd.Series(output, index=prices.index, name="ema")

# Todays True Range
def ttr(data: pd.DataFrame) -> pd.Series:
    output = np.full(data.shape[0], np.nan)
    for positions in data.groupby('ticker').indices.values():
        current = data.iloc[positions].copy()
        current['previous_close'] = current['close'].shift(-1)
        tr = current.apply(
            lambda row: np.max([row["high"], row["previous_close"]])
            - np.min([row["low"], row["previous_close"]]), axis=1
        )
        output[positions] = tr
    return pd.Series(output, index=data.index, name="ttr")
        

