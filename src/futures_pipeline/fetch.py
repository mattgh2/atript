#!usr/bin/python3
from .datareader import (
    load_prior_data,
    fetch_latest,
    fetch_lookback,
    fetch_range,
    fetch_training_set
)
from pathlib import Path
import pandas as pd
from .validate import validate_data
from massive import RESTClient
from .typedefs import (
    MassiveParameters,
    FetchLookbackArgs,
    FetchRangeArgs,
    FetchLatestArgs,
    FetchContextArgs,
    FetchTrainArgs,
)
from .config import TRAINING_DATA_DIR, RAW_DATA_DIR

# VWAP, RSI, EMA, ATR, bollinger Bands (20, 2\sigma), MACD


def fetch_context(
    massive_params: MassiveParameters,
    massive_client: RESTClient,
    args: FetchContextArgs,
) -> None:
    match args:
        case FetchLatestArgs():
            data = fetch_latest(massive_params, massive_client)
        case FetchLookbackArgs():
            data = fetch_lookback(
                args.period, args.depth, massive_params, massive_client
            )
        case FetchRangeArgs():
            data = fetch_range(args.begin, args.end, massive_params, massive_client)

    print(f"Fetched {data.shape[0]:,} observations for {args.symbol}")

    data = validate_data(list(data.to_dict(orient='index').values()))

    # Write data to parquete files.
    data_dir = RAW_DATA_DIR / f"{args.symbol}-{args.resolution}"
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.Series(data["session_end_date"], dtype="datetime64[ns]")
    for day, rows in data.groupby(dates.dt.date):
        prior: pd.DataFrame = load_prior_data(args.symbol, args.resolution, day, day, data_dir)

        if not prior.empty:
            rows = pd.concat([rows, prior], ignore_index=True).drop_duplicates(
                subset="window_start"
            )

        path: str = f"{data_dir}/{args.symbol}-{args.resolution}-{day}.parquet"
        rows.to_parquet(path, index=False)
        print(f"Wrote {path} ({rows.shape[0]:,} rows)")


def fetch_train(massive_client: RESTClient, args: FetchTrainArgs):

    data: pd.DataFrame = fetch_training_set(
        args.symbol, args.resolution, massive_client, args.from_date, args.years
    )

    data_dir = TRAINING_DATA_DIR / f"{args.symbol}-{args.resolution}"
    data_dir.mkdir(parents=True, exist_ok=True)

    data = validate_data(list(data.to_dict(orient='index').values()))

    output_file = f"{args.symbol}-{args.resolution}-train.parquet"
    data.to_parquet(data_dir / output_file)

