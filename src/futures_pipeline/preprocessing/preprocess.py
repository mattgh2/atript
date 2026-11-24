import pandas as pd
from ..datareader import load_prior_data
from ..datareader.readerutil import fetch_session_hours, get_product_code, fetch_session_dates
from pathlib import Path
from ..config import PROCESSED_DATA_DIR, RAW_DATA_DIR, TRAINING_DATA_DIR, WEB_PATH
from ..typedefs import CandleResolution
import numpy as np
from .indicators.momentum import smoothed_rsi, RoC, fast_k, slow_k, cci
from .indicators.trend import ema, dma, sma, t3ma, signed_efficiency_ratio
from .indicators.volatility import bollinger_bands, rolling_std, atr, vr, kc
from .indicators.volume import vwap
from .preproc_util import get_returns
from datetime import timedelta, datetime, date, time
import pandas_market_calendars as mcal
from shutil import rmtree

def preprocess(symbol: str, resolution: str, train: bool) -> None:

    processed_path: Path = Path(PROCESSED_DATA_DIR) / f"{symbol}-{resolution}"
    raw_dir: Path | None = None

    if not train:
        raw_dir = RAW_DATA_DIR 

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
            "transactions",
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
        data.drop_duplicates(subset=['ticker', "real_timestamp"])
        .sort_values("real_timestamp", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    candle_resolution: CandleResolution = CandleResolution(resolution)

    # data["returns"] = get_returns(data["close"])

    # Creates rows: log_elapsed_intervals, has_time_gap
    data = get_time_features(data, candle_resolution)
    data = fill_missing(data, candle_resolution)
    data = get_time_features(data, candle_resolution)

    data["_session_key"] = pd.to_datetime(
        data["session_end_date"]
    ).dt.strftime("%Y-%m-%d")

    session_audit = audit_session_coverage(
        data,
        get_product_code(symbol),
        candle_resolution,
    )

    audit_path = WEB_PATH / f"audit.html"
    if not audit_path.exists():
        audit_path.touch()

    with open(audit_path, 'w') as fp:
        fp.write(str(session_audit))

    # A first/last session of a contract cannot be reliably padded on its
    # unobserved outer side. Keep it as context/training data, but do not
    # score its targets when the source session was incomplete.
    is_contract_edge_session = (
        data.groupby("ticker")["_session_key"]
        .transform(lambda sessions: sessions.eq(sessions.min()) | sessions.eq(sessions.max()))
    )

    data = data.merge(
        session_audit,
        on=["ticker", "_session_key"],
        how="left",
        validate="many_to_one",
    )

    data["eval_target_ok"] = (
        data["was_missing"].eq(0) & ~(is_contract_edge_session & ~data["session_complete"])
    ).astype("int8")

    data = data.drop(columns=["_session_key"])

    data["prev_was_missing"] = data.groupby('ticker')['was_missing'].shift(-1)
    data['follows_missing'] = (data['prev_was_missing'].eq(1) & data['was_missing'].eq(0)).astype("int8")
    data = data.drop(columns=['prev_was_missing'], axis=1)

    data = data.set_index("ticker")

    minutes = (
            data['real_timestamp'].dt.hour * 60
            + data['real_timestamp'].dt.minute
    )

    angle = 2 * np.pi * minutes / 1440

    data['cosine_time'] = np.cos(angle)
    data['sine_time'] = np.sin(angle)

    # Get the NYSE calendar
    nyse = mcal.get_calendar("NYSE")

    # For each session_end date, get the opening and closing of the NYSE.
    schedule: pd.DataFrame = nyse.schedule(
        start_date=data["session_end_date"].min(),
        end_date=data["session_end_date"].max(),
    )

    schedule = (
        schedule.rename_axis("session_end_date")
        .rename(columns={"market_open": "cash_open", "market_close": "cash_close"})
    )
    schedule.index = schedule.index.strftime('%Y-%m-%d') # type: ignore

    data = data.join(
            schedule,
            on='session_end_date',
            validate='many_to_one'
    )

    for column in ('cash_open', 'cash_close'):
        data[column] = pd.to_datetime(data[column], utc=True)

    data["is_cash_session"] = ((data["cash_open"] <= data["real_timestamp"]) & (
        data["cash_close"] > data["real_timestamp"]
    )).astype('int8')

    # Minutes since cash open / until close
    data['minutes_since_cash_open'] = np.where(
            data['is_cash_session'].eq(1),
            (data['real_timestamp'] - data['cash_open']).dt.total_seconds() / 60,
            0
    )
    data['minutes_until_cash_close'] = np.where(
            data['is_cash_session'].eq(1),
            (data['cash_close'] - data['real_timestamp']).dt.total_seconds() / 60,
            0
    )

    data['log_volume'] = np.log1p(data['volume'])
    data['log_transactions'] = np.log1p(data['transactions'])

    data['ATR'] = atr(data)
    data['normalized_atr'] = atr(data) / data['close']
    data['VR'] = vr(data)

    previous = data.groupby(['ticker'])['close'].shift(-1)
    previous_5 = data.groupby(['ticker'])['close'].shift(-5)

    data['log_returns_1'] = np.log(data['close'] / previous)
    data['log_returns_5'] = np.log(data['close'] / previous_5)

    data["ema_40"] = ema(data["close"], period=40)

    ema_slope_atr = np.full(len(data), np.nan)
    np.divide(
        data["ema_40"] - data.groupby("ticker", sort=False)["ema_40"].shift(-5),
        5 * data["ATR"],
        out=ema_slope_atr,
        where=data['ATR'].ne(0)
    )
    data['ema_slope_atr'] = ema_slope_atr

    ema_spread_atr = np.full(len(data), np.nan)
    np.divide(
        ema(data['close'], period=10) - data['ema_40'],
        data["ATR"],
        out=ema_spread_atr,
        where=data['ATR'].ne(0)
    )

    data['ema_spread_atr'] = ema_spread_atr

    data['signed_efficency_ratio'] = signed_efficiency_ratio(data['close'])

    # signed candle body is C_t - O_t
    data['signed_candle_body_atr'] = (
            (data['close'] - data['open'])
            / data['ATR'].where(data['ATR'].gt(0))
    )

    candle_range = data['high'] - data['low']

    data['close_location_within_candle_range'] = (
            (2 * data['close'] - data['high'] - data['low'])
            / candle_range.where(candle_range.gt(0))
    )

    # Percentage distance between the close and its EMA / VWAP. More useful for forecasting returns.
    data["VWAP"] = vwap(data)
    data['close_to_vwap'] = data['close'] / data['VWAP'] - 1
    data["close_to_vwap_atr"] = (
        (data["close"] - data["VWAP"])
        / data["ATR"].where(data["ATR"].gt(0))
    )

    # data.loc[missing_prev, 'returns'] = np.nan

    # # Momentum
    # data["RSI"] = smoothed_rsi(data["close"])
    # data['FSO'] = fast_k(data)
    # data['SSO'] = slow_k(data['FSO'])
    # data['CCI'] = cci(data)
    # data['ROC'] = RoC(data['close'])
    #
    # # Volume.
    #
    # # Trend indicators.
    # data['SMA'] = sma(data['close'])
    # data['DMA'] = dma(data['SMA'])
    # data['T3MA'] = t3ma(data['close'])
    #
    # # Volatility.
    # # data['rolling_std_20'] = rolling_std(data['returns'], 20)
    # bb = bollinger_bands(data['close'])
    # keltner = kc(data)
    # data["percent_b"] = bb.percent_b
    # data["bollinger_bandwidth"] = (bb.upper_band - bb.lower_band) / bb.middle_band
    #
    # keltner_position = np.full(len(data), np.nan)
    # np.divide(
    #     (data["close"] - keltner.lower_band),
    #     keltner.upper_band - keltner.lower_band,
    #     out=keltner_position,
    #     where=(keltner.upper_band - keltner.lower_band) > 0
    # )
    #
    # data["keltner_position"] = keltner_position
    #
    # keltner_width = np.full(len(data), np.nan)
    # np.divide(
    #         keltner.upper_band - keltner.lower_band,
    #         keltner.middle_band,
    #         out=keltner_width,
    #         where=keltner.middle_band > 0
    # )
    # data['keltner_width'] = keltner_width
    #
    # data['bollinger_squeeze'] = keltner.bollinger_squeeze
    # data['bollinger_squeeze_ratio'] = keltner.squeeze_ratio
    #

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.reset_index()

    # prior: pd.DataFrame = load_prior_data(symbol, resolution, data_dir=PROCESSED_DATA_DIR)

    # data = pd.concat([data, prior], ignore_index=True).drop_duplicates(['ticker', 'real_timestamp'])

    data = data.sort_values(
        "real_timestamp", ascending=False, kind="stable"
    ).reset_index(drop=True)

    # Generates synthetic timestamps begining on 2000-01-01.
    data = convert_timestamps(data, candle_resolution)

    if processed_path.exists():
        rmtree(processed_path)
    processed_path.mkdir(exist_ok=True, parents=True)

    # Write data to parquete files.
    dates = pd.Series(data["session_end_date"], dtype="datetime64[ns]")
    for day, rows in data.groupby(dates.dt.date):
        path: str = f"{processed_path}/{symbol}-{resolution}-{day}.parquet"
        rows.to_parquet(path, index=False)
        print(f"Wrote {path} ({rows.shape[0]:,} rows)")


def convert_timestamps(df: pd.DataFrame, resolution: CandleResolution) -> pd.DataFrame:

    model_step = df.groupby("ticker").cumcount(ascending=False)

    start = pd.Timestamp("2000-01-01")
    df["model_timestamp"] = start + pd.to_timedelta(model_step * resolution.length, unit=resolution.to_timedelta_unit())

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

"""
Bound to the start or the end of a session depending
on whether the first candle is missing or the last candle is missing
If last, then current = min(prev_session_end, current). If first, then prev = max(current_session_start, prev)
"""
def fill_missing(data: pd.DataFrame, resolution: CandleResolution):

    step = pd.Timedelta(
            resolution.length,
            unit=resolution.to_timedelta_unit()
    )
    # Fill missing rows with nan.
    missing_rows: list[pd.Series] = []
    for ticker, contract_df in data.groupby('ticker', sort=False):
        first = contract_df['real_timestamp'].min()
        last = contract_df['real_timestamp'].max()
        observed = pd.DatetimeIndex(contract_df['real_timestamp'])

        hours = fetch_session_hours(
                get_product_code(str(ticker)),
                contract_df['session_end_date'].min(),
                contract_df['session_end_date'].max(),
        )

        for session_date, intervals in hours.items():
            for opened, closed in intervals:
                if closed <= first or opened > last:
                    continue
        
                expected = pd.date_range(
                        opened,
                        closed,
                        freq=step,
                        inclusive='left'
                )
                expected = expected[(expected >= first) & (expected <= last)]

                missing = expected.difference(observed)
                missing_rows.extend(
                        pd.Series({
                            'ticker': ticker,
                            'real_timestamp': timestamp,
                            'session_end_date': session_date,
                            'was_missing': 1,
                            'has_time_gap': 0,
                        })
                        for timestamp in missing
                )

    missing_df = pd.DataFrame(missing_rows)

    data['was_missing'] = 0

    data = (
        pd.concat([data, missing_df], ignore_index=True)
        .sort_values("real_timestamp", ascending=False)
        .reset_index(drop=True)
    )

    data['was_missing'] = data['was_missing'].astype(np.int8)
    data['has_time_gap'] = data['has_time_gap'].astype(np.int8)

    return data


def audit_session_coverage(
    data: pd.DataFrame,
    product_code: str,
    resolution: CandleResolution,
) -> pd.DataFrame:
    """Return observed/synthetic coverage for each ticker/session."""
    interval = pd.Timedelta(
        resolution.length,
        unit=resolution.to_timedelta_unit(),
    )

    work = data.copy()
    work["_session_key"] = pd.to_datetime(
        work["session_end_date"]
    ).dt.strftime("%Y-%m-%d")

    session_hours = fetch_session_hours(
        product_code,
        work["_session_key"].min(),
        work["_session_key"].max(),
    )

    audit_rows: list[dict] = []

    for (ticker, session_key), group in work.groupby(
        ["ticker", "_session_key"],
        sort=False,
    ):
        intervals = session_hours.get(str(session_key))
        if not intervals:
            audit_rows.append(
                {
                    "ticker": ticker,
                    "_session_key": session_key,
                    "expected_bars": np.nan,
                    "observed_bars": np.nan,
                    "synthetic_bars": np.nan,
                    "unrepresented_bars": np.nan,
                    "session_complete": False,
                    "session_covered": False,
                }
            )
            continue

        expected = pd.DatetimeIndex(
            np.concatenate(
                [
                    pd.date_range(
                        start=opened_at,
                        end=closed_at,
                        freq=interval,
                        inclusive="left",
                    ).to_numpy()
                    for opened_at, closed_at in intervals
                ]
            )
        )

        observed_timestamps = group.loc[
            group["was_missing"].eq(0),
            "real_timestamp",
        ]

        synthetic_timestamps = group.loc[
            group["was_missing"].eq(1),
            "real_timestamp",
        ]

        observed = observed_timestamps.isin(expected).sum()
        synthetic = synthetic_timestamps.isin(expected).sum()
        represented = pd.Index(
            pd.concat([observed_timestamps, synthetic_timestamps]).unique()
        ).isin(expected).sum()

        audit_rows.append(
            {
                "ticker": ticker,
                "_session_key": session_key,
                "expected_bars": len(expected),
                "observed_bars": observed,
                "synthetic_bars": synthetic,
                "unrepresented_bars": len(expected) - represented,
                # Strict source-data completeness:
                "session_complete": observed == len(expected),
                # The padded series represents every expected bar:
                "session_covered": represented == len(expected),
            }
        )

    return pd.DataFrame(audit_rows)
