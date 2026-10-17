import pandas as pd
from ..datareader import load_prior_data
from ..datareader.readerutil import fetch_session_hours, get_product_code, fetch_session_dates
from pathlib import Path
from ..config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TRAINING_DATA_DIR
from ..typedefs import CandleResolution
import numpy as np
from .indicators.momentum import smoothed_rsi, RoC, fast_k, slow_k, cci
from .indicators.trend import ema, dma, sma, t3ma
from .indicators.volatility import bollinger_bands, rolling_std, atr, vr, kc
from .indicators.volume import vwap
from .preproc_util import get_returns
from datetime import timedelta, datetime, date, time
import pudb

def preprocess(symbol: str, resolution: str, train: bool) -> None:

    processed_path: Path = Path(PROCESSED_DATA_DIR) / f"{symbol}-{resolution}"
    processed_path.mkdir(exist_ok=True, parents=True)

    raw_dir: Path | None = None

    if not train:
        raw_dir = RAW_DATA_DIR / f"{symbol}-{resolution}"

    data: pd.DataFrame = load_prior_data(symbol, resolution, data_dir=raw_dir, train=train)

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

    # data["returns"] = get_returns(data["close"])

    # Creates rows: log_elapsed_intervals, has_time_gap
    data = get_time_features(data, candle_resolution)


    # Need this column for fill_missing()
    data['missing_prev'] = get_missing_gaps(data, symbol, candle_resolution)
    data = fill_missing(data, candle_resolution)

    data = data.drop(columns=['missing_prev'],axis=1)
    data['follows_close'] = get_time_features(data, candle_resolution)

    data = data.set_index("ticker")

    # data.loc[missing_prev, 'returns'] = np.nan

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
        path: str = f"{processed_path}/{symbol}-{resolution}-{day}.parquet"
        prior: pd.DataFrame = load_prior_data(symbol, resolution, day, day, processed_path)
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

    # This will fail for 2025-01-09, Going to need to drop the session.
    if missing := set(data['session_end_date'].unique()) - set(session_hours):
        raise ValueError(f"Missing session schedules for: {sorted(missing)}")

    trading_intervals = sorted(
            (opened_at, closed_at)
            for hours in session_hours.values()
            for opened_at, closed_at in hours
    )

    interval = pd.to_timedelta(
        candle_resolution.length, unit=candle_resolution.to_timedelta_unit()
    )

    def contains_missing_candle(older: pd.Timestamp, current: pd.Timestamp) -> bool:
        # 2026-09-01 17:00:00-05:00 
        target = pd.Timestamp(datetime.combine(
                date.fromisoformat("2026-09-01"),
                time(17,0,0)
        ),tz="America/Chicago") 

        # if current == target:
        #     breakpoint()

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

"""
Bound to the start or the end of a session depending
on whether the first candle is missing or the last candle is missing
If last, then current = min(prev_session_end, current). If first, then prev = max(current_session_start, prev)
"""
def fill_missing(data: pd.DataFrame, resolution: CandleResolution):
    if 'missing_prev' not in data.columns:
        raise ValueError('Missing missing_prev column.')

    # Fill missing rows with nan.
    missing_rows: list[pd.Series] = []
    for ticker, df in data.groupby('ticker', sort=False):
        df['prev_timestamp'] = df['real_timestamp'].shift(-1)
        df['prev_session'] = df['session_end_date'].shift(-1)

        for idx, row in df.iterrows():
            if not row['missing_prev']:
                continue

            prev_timestamp = pd.Timestamp(row['prev_timestamp'])
            current_timestamp = pd.Timestamp(row['real_timestamp'])

            _, prev_session_close = fetch_session_dates(
                row["ticker"], row["prev_session"], row["prev_session"]
            )[row["prev_session"]]

            current_session_open, current_session_close = fetch_session_dates(
                row["ticker"], row["session_end_date"], row["session_end_date"]
            )[row["session_end_date"]]


            candle_length_seconds: int = int(
                pd.to_timedelta(
                    resolution.length, unit=resolution.to_timedelta_unit()
                ).total_seconds()
            )

            prev_end = prev_timestamp
            current_begin = current_timestamp

            # Handles case where the current timestamp is the first candle of a session,
            # thus previous timestamp is from the previous session.
            # Avoids creating expected rows during a scheduled close.

            # Missing candles are either the first or last in a session.
            if prev_session_close != current_session_close:

                # Missing both the prev sessions final candles and the current sessions opening candles.
                if current_timestamp > current_session_open and prev_timestamp < prev_session_close:
                    prev_end = prev_session_close
                    current_begin = current_session_open - timedelta(seconds=candle_length_seconds)

                # If the current timestamp is not the opening candle,
                # then the current sessions starting candles are missing.
                elif current_timestamp > current_session_open:
                    current_begin = current_session_open

                # The prev sessions final candles must be missing.
                else:
                    prev_end = prev_session_close

            if pd.isna(prev_timestamp) or pd.isna(current_timestamp):
                raise ValueError("Gap has a missing timestamp.")
            if current_timestamp <= prev_timestamp:
                raise ValueError("Gap timestamps must be strictly ordered.")

            missing_left = generate_missing_rows(
                prev_timestamp, prev_end, candle_length_seconds, row['prev_session'], row['ticker']
            )
            missing_right = generate_missing_rows(
                current_begin, current_timestamp, candle_length_seconds, row['session_end_date'], row['ticker']
            )

            missing_rows.extend((*missing_left, *missing_right))

    missing_df = pd.DataFrame(missing_rows)

    data = (
        pd.concat([data, missing_df], ignore_index=True)
        .sort_values("real_timestamp", ascending=False)
        .reset_index(drop=True)
    )
    data['was_missing'] = data['was_missing'].fillna(0)
    return data

def generate_missing_rows(start, stop, interval_len, session_end_date, ticker):
    missing_rows = []
    seconds_diff = int((stop - start).total_seconds())

    missing_timestamps = (
        start + timedelta(seconds=offset)
        for offset in range(
            interval_len, seconds_diff, interval_len
        )
    )

    for ts in missing_timestamps:
        missing_row = pd.Series(
            [session_end_date, ts, ticker, 0, 1],
            index=["session_end_date", "real_timestamp", "ticker", "has_time_gap", "was_missing"],
        )
        missing_rows.append(missing_row)
    return missing_rows
