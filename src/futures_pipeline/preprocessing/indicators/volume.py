import pandas as pd
import numpy as np
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
