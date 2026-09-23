import pandas as pd
from ..datareader import load_prior_data
from ..datareader.readerutil import fetch_session_hours, get_product_code
from pathlib import Path
from ..config import PROCESSED_DATA_DIR
from ..typedefs import CandleResolution
import numpy as np
from .indicators.momentum import smoothed_rsi, RoC, fast_k, slow_k, cci
from .indicators.trend import ema, dma, sma, t3ma
from .indicators.volatility import bollinger_bands, rolling_std, atr, vr, kc
from .indicators.volume import vwap
from .preproc_util import get_returns

def preprocess(symbol: str, resolution: str, data_dir: Path, train: bool) -> None:
    processed_path: Path = Path(PROCESSED_DATA_DIR) / symbol
    processed_path.mkdir(exist_ok=True, parents=True)

    data: pd.DataFrame = load_prior_data(data_dir, symbol, train=train)

    if data.empty:
        print(f"No data available for {symbol}.")
        return

    print(f"Processing {data.shape[0]} records for {symbol}.")

    raw_columns = [
            "window_start",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "session_end_date"
    ]

    missing_cols = set(raw_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}.")

    data['window_start'] = pd.to_datetime(
            data['window_start'],
            utc=True,
            errors="coerce"
    ).dt.tz_convert("America/Chicago")

    missing_raw = data[raw_columns].isna().any(axis=1)
    if missing_raw.any():
        raise ValueError(f"Found {missing_raw.sum()} incomplete raw rows.")

    data['real_timestamp'] = data['window_start']
    data = data.drop(['window_start'], axis=1)

    data = (
        data.drop_duplicates(subset=["real_timestamp"])
        .sort_values("real_timestamp", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


    candle_resolution: CandleResolution = CandleResolution(resolution)

    data = get_time_features(data, candle_resolution)

    data = data.set_index("ticker")
    data["returns"] = get_returns(data["close"])

    missing_prev = get_missing_gaps(data, symbol, candle_resolution)
    data.loc[missing_prev, 'returns'] = np.nan

    # Momentum
    data["RSI"] = smoothed_rsi(data["close"])
    data['FSO'] = fast_k(data)
    data['SSO'] = slow_k(data['FSO'])
    data['CCI'] = cci(data)
    data['ROC'] = RoC(data['close'])

    # Volume.
    data["VWAP"] = vwap(data)

    # Trend indicators.
    data['SMA'] = sma(data['close'])
    data["EMA"] = ema(data["close"])
    data['DMA'] = dma(data['SMA'])
    data['T3MA'] = t3ma(data['close'])

    # TODO: These need to be grouped by ticker
    # data['close_to_ema'] = data['close'] / data["ema"] - 1
    # data['ema_slope'] = data['ema'] / data['ema'].shift(-1) - 1
    # data['ema_slope_4'] = data['ema'] / data['ema'].shift(-4) - 1

    # Volatility.
    bb = bollinger_bands(data['close'])
    keltner = kc(data)
    data['rolling_std_20'] = rolling_std(data['returns'], 20)
    data["percent_b"] = bb.percent_b
    data["upper_b"] = bb.upper_band
    data["lower_b"] = bb.lower_band
    data['lower_kc'] = keltner.lower_band
    data['upper_kc'] = keltner.upper_band
    data['bollinger_squeeze'] = keltner.bollinger_squeeze
    data['ATR'] = atr(data)
    data['VR'] = vr(data)

    # TODO: Group by ticker
    # data['intrabar_range'] = (data['high'] - data['low']) / data['close']
    # previous_close = data['close'].shift(-1)
    # data['true_range'] = pd.concat(
    #         [
    #             data['high'] - data['low'],
    #             (data['high'] - previous_close).abs(),
    #             (data['low'] - previous_close).abs()
    #         ], axis=1
    # ).max(axis=1)
    # data['normalized_true_range'] = data["true_range"] / data['close']

    # Percentage distance between the close and its EMA / VWAP. More useful for forecasting returns.
    # TODO: Group by ticker
    # data['close_to_vwap'] = data['close'] / data['VWAP'] - 1

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.reset_index()

    # Chronos-2 supports nan values.
    # data = data.dropna(subset=model_features).reset_index(drop=True)

    data = convert_timestamps(data, candle_resolution)

    # Write data to parquete files.
    dates = pd.Series(data["session_end_date"], dtype="datetime64[ns]")
    for day, rows in data.groupby(dates.dt.date):
        path: str = f"{processed_path}/{symbol}-{day}.parquet"
        prior: pd.DataFrame = load_prior_data(processed_path, symbol, day, day)
        if not prior.empty:
            rows = pd.concat([rows, prior], ignore_index=True).drop_duplicates(
                subset="real_timestamp"
            )
        rows = rows.sort_values(
            "real_timestamp", ascending=False, kind="stable"
        ).reset_index(drop=True)
        rows.to_parquet(path, index=False)
        print(f"Wrote {path} ({rows.shape[0]:,} rows)")


"""

"""
def convert_timestamps(df: pd.DataFrame, resolution: CandleResolution) -> pd.DataFrame:

    model_step = df.groupby("ticker").cumcount(ascending=False)

    start = pd.Timestamp("2000-01-01")
    df["model_timestamp"] = start + pd.to_timedelta(model_step * resolution.length, unit=resolution.to_timedelta_unit())  # type: ignore

    return df


# Covariate features to preserve  market-time information.
def get_time_features(df: pd.DataFrame, resolution: CandleResolution) -> pd.DataFrame:
    interval_length = pd.Timedelta(
        resolution.length, unit=resolution.to_timedelta_unit()
    )

    elapsed_intervals = (
        ((df["real_timestamp"] - df.groupby("ticker")["real_timestamp"].shift(-1)) / interval_length)  # type: ignore
        .fillna(1.0)
        .astype("float32")
    )
    df["has_time_gap"] = (elapsed_intervals > 1.0).astype("int8")

    # Intervals are very skewed, compute their log.
    log_interval = np.log(elapsed_intervals)
    df['log_elapsed_intervals'] = log_interval

    return df

def get_missing_gaps(data: pd.DataFrame, symbol: str, candle_resolution: CandleResolution) -> np.ndarray:

    prev_observed = data.groupby('ticker', sort=False)['real_timestamp'].shift(-1)

    session_hours: dict[str, list] = fetch_session_hours(
        get_product_code(symbol),
        data["session_end_date"].min(),
        data["session_end_date"].max(),
    )
    if missing := set(data['session_end_date'].unique()) - set(session_hours):
        raise ValueError(f"Missing session schedules for: {sorted(missing)}")

    trading_intervals = sorted(
            (opened_at, closed_at)
            for hours in session_hours.values()
            for opened_at, closed_at in hours
    )

    interval =  pd.to_timedelta(
        candle_resolution.length, unit=candle_resolution.to_timedelta_unit()
    )

    def contains_missing_candle(older: pd.Timestamp, current: pd.Timestamp) -> bool:
        for opened_at, closed_at in trading_intervals:
            if closed_at <= older or opened_at >= current:
                continue
            if older < opened_at:
                candidate = opened_at
            else:
                steps = ((older - opened_at) // interval) + 1
                candidate = opened_at + steps * interval
            if candidate < current and candidate < closed_at:
                return True

        return False

    missing_prev = np.zeros(len(data), dtype=bool)
    gap_positions = np.flatnonzero(
            data['has_time_gap'].to_numpy(dtype=bool)
    )

    for position in gap_positions:
        older = prev_observed.iloc[position]
        current = data['real_timestamp'].iloc[position]
        if pd.notna(older):
            missing_prev[position] = contains_missing_candle(older,current)

    return missing_prev
