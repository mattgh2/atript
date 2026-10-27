import pandas as pd
import numpy as np

"""
Calculates VWAP.

@param

@note VWAP resets at the start of each session.

"""
def vwap(df: pd.DataFrame) -> pd.Series: 
    data = df.copy()

    data['price'] = (df['high'] + df['low'] + df['close']) / 3
    data["_position"] = np.arange(data.shape[0])

    data = data.sort_values("real_timestamp", kind='stable')

    # A missing OHLCV row is a break in the cumulative calculation, rather
    # than an observation to skip.  This prevents post-gap VWAP values from
    # incorporating prices or volume from before the gap.
    valid = data[['high', 'low', 'close', 'volume']].notna().all(axis=1)
    ticker = (
        data['ticker']
        if 'ticker' in data.columns
        else pd.Series(data.index, index=data.index, name='ticker')
    )

    session_keys = [data['session_end_date'], ticker]
    segment = (~valid).groupby(session_keys, sort=False).cumsum()

    price_volume = (data['price'] * data['volume']).where(valid)
    volume = data['volume'].where(valid)
    segment_keys = [*session_keys, segment]

    cumulative_price_volume = price_volume.groupby(segment_keys, sort=False).cumsum()
    cumulative_volume = volume.groupby(segment_keys, sort=False).cumsum()

    data['VWAP'] = (
        cumulative_price_volume / cumulative_volume
    ).where(valid & cumulative_volume.gt(0)).to_numpy()

    vwap = data.sort_values('_position')['VWAP'].to_numpy()
    return pd.Series(vwap, index=df.index, name="VWAP")
