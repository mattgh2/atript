#!usr/bin/python3
from .datareader import (
    load_prior_data,
    fetch_latest,
    fetch_lookback,
    fetch_range,
)
from pathlib import Path
from collections import defaultdict
import pandas as pd
from itertools import chain
from .config import RAW_DATA_DIR
from .validate import validate_data
from massive import RESTClient
from .typedefs import MassiveParameters, FetchLookbackArgs, FetchRangeArgs, FetchLatestArgs, FetchArgs

# VWAP, RSI, EMA, ATR, bollinger Bands (20, 2\sigma), MACD

def fetch(massive_params: MassiveParameters, massive_client: RESTClient, args: FetchArgs) -> None:

    Path(f"{RAW_DATA_DIR}/{args.ticker}").mkdir(parents=True, exist_ok=True)
    match args:
        case FetchLatestArgs():
            data = fetch_latest(massive_params, massive_client)
        case FetchLookbackArgs():
            data = fetch_lookback(
                args.period, args.depth, massive_params, massive_client
            )
        case FetchRangeArgs():
            data = fetch_range(args.begin, args.end, massive_params, massive_client)
        case _:
            return
    # Check if data is empty.
    try:
        first = next(iter(data))
    except StopIteration:
        return

    data = chain((first,), data)
    data = validate_data(data)

    # Convert data into DataFrame
    d: defaultdict[str, list] = defaultdict(lambda: [])
    for futuresagg in data:
        for k, v in (vars(futuresagg)).items():
            d[k].append(v)
    df: pd.DataFrame = pd.DataFrame(d)

    print(f"Fetched {df.shape[0]:,} observations for {args.ticker}")

    # Write data to parquete files.
    dates = pd.Series(df["session_end_date"], dtype="datetime64[ns]")
    for day, rows in df.groupby(dates.dt.date):
        prior: pd.DataFrame | None = load_prior_data(Path(RAW_DATA_DIR) / args.ticker, args.ticker, day, day)

        if prior is not None:
            rows = pd.concat([rows, prior], ignore_index=True).drop_duplicates(
                subset="window_start"
            )

        path: str = f"{RAW_DATA_DIR}/{args.ticker}/{args.ticker}-{day}.parquet"
        rows.to_parquet(path, index=False)
        print(f"Wrote {path} ({rows.shape[0]:,} rows)")
