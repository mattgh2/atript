import pandas as pd
from ..datareader import load_prior_data, load_train
from pathlib import Path
from ..config import PROCESSED_DATA_DIR
from ..typedefs import CandleResolution
import numpy as np
from .indicators import (
    smoothed_rsi,
    percent_b,
    get_returns,
    sma,
    ema,
    msi,
    vwap,
    wma,
    vwap,
    ema,
    rolling_std
)


def preprocess(symbol: str, resolution: str, data_dir: Path, train: bool = False) -> None:
    processed_path: Path = Path(PROCESSED_DATA_DIR) / symbol
    processed_path.mkdir(exist_ok=True, parents=True)

    data: pd.DataFrame | None = load_prior_data(data_dir, symbol, train=train)

    if data is None:
        print(f"No data available for {symbol}.")
        return

    print(f"Processing {data.shape[0]} records for {symbol}.")



    # prior_n: pd.DataFrame | None = load_last_n(
    #     raw_path, ticker, settings.indicator_lookback
    # )

    # Concatenate new data with the prior n observations needed to calculate indicators.
    # if prior_n is not None:
    #     data = pd.concat([data, prior_n], ignore_index=True)

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
    model_features = [
            "returns",
            "volume",
            "percent_b",
            "rsi",
            "close_to_ema",
            "close_to_vwap",
            "has_time_gap",
            "log_elapsed_intervals"
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

    # Set return calcuated between missing observations within the same session to nan.
    previous = data.groupby('ticker')['session_end_date'].shift(-1)
    missing_within_session = (
            data['has_time_gap'].eq(1) & data['session_end_date'].eq(previous)
    )
    data.loc[missing_within_session, 'returns'] = np.nan


    data = data.set_index("ticker")

    data["returns"] = get_returns(data["close"])

    data["rsi"] = smoothed_rsi(data["close"])
    data["percent_b"] = percent_b(data["close"])
    data["VWAP"] = vwap(data)

    # Trend indicators.
    data["ema"] = ema(data["close"])

    # TODO: These need to be grouped by tickerjk
    # data['close_to_ema'] = data['close'] / data["ema"] - 1
    # data['ema_slope'] = data['ema'] / data['ema'].shift(-1) - 1
    # data['ema_slope_4'] = data['ema'] / data['ema'].shift(-4) - 1

    # Volatility.
    data['rolling_std_20'] = rolling_std(data['returns'], 20)

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
        prior: pd.DataFrame | None = load_prior_data(processed_path, symbol, day, day)
        if prior is not None:
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
