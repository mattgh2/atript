import pandas as pd
import numpy as np

"""
Computes the returns for a sequence of candlestick closes.
@param close An array of closing values for candlesticks.

@Note Returns are the relative change in closing value between observation k and k + 1.
@Note Assumes close is ordered by most recent observation to least recent observation.
"""
def get_returns(close: pd.Series) -> pd.Series:
    returns = np.full(len(close), np.nan)
    for positions in close.groupby(level=0).indices.values():
        symbol_close = close.iloc[np.array(positions)]
        returns[positions[:-1]] = -np.diff(symbol_close) / symbol_close.iloc[1:]
    return pd.Series(returns, index=close.index, name="returns")
