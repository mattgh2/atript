from massive.rest.futures import FuturesAgg, FuturesContract, FuturesProduct
from massive import RESTClient
from copy import copy
from .typedefs import MassiveParameters, ContractSpec
from collections.abc import Iterator
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta
from .config import RAW_DATA_DIR
from typing import cast
import re

"""
Fetches OHLC data from Massive.com.

@param client A massive Restclient instance.

@api-param limit The number of results to return per page.
@api-param window_start The start time of the candlesticks. A Unix timestamp or YYYY-MM-DD value.
@api-param resolution The size of each aggregate candle, specified as a number followed 
                  by a unit: sec, min, hour, week, month, quarter, year.
@api-param sort Sort results by field and direction using dotted notation e.g. ticker.asc, name.desc

@return  OHLC data for each provided ticker.
"""
def fetch_data(
    client: RESTClient, fetch_parameters: MassiveParameters
) -> Iterator[FuturesAgg]:
    data = cast(
        Iterator[FuturesAgg],
        client.list_futures_aggregates(**fetch_parameters, raw=False),
    )
    try:
        first = next(data)
    except StopIteration:
        print("No data Returned")
        return
    yield first
    yield from data


def fetch_contract_spec(ticker: str, client: RESTClient) -> ContractSpec:
    contract: FuturesContract | None = cast(FuturesContract | None, next(client.list_futures_contracts(
        ticker=ticker, limit=1
    ), None))


    if contract is None:
        raise ValueError(f"Failed to fetch contract for {ticker}.")

    match = re.match(r"([A-Z0-9]+?)[FGHJKMNQUVXZ]\d{1,4}", ticker)
    if not match:
        raise ValueError(f"Error parsing product name from {ticker}.")
    product_code = match.group(1)

    product: FuturesProduct | None = cast(
        FuturesProduct | None,
        next(client.list_futures_products(product_code=product_code, limit=1, raw=False), None),
    )

    if product is None:
        raise ValueError(f"Failed to fetch product for {ticker}")

    multiplier: float | None = product.unit_of_measure_qty
    tick_size: float | None = contract.trade_tick_size

    if multiplier is None or tick_size is None:
        raise ValueError(
            f"Failed to fetch contract {"multiplier" if multiplier is None else "tick_size"}."
        )

    price_per_tick = multiplier * tick_size

    return ContractSpec(
            contract=ticker,
            tick_size=tick_size,
            price_per_tick=price_per_tick,
            multiplier=multiplier
    )

"""
Loads stored data from disk from a range of dates

@param ticker The ticker to retreive.
@param begin The first trading day  to retrieve.
@param end The last trading day  to retrieve.

@note If end is omited, the function will return all data 
      from the start date through the most recent stored trading day.
      If begin is omited, the function will return all data from 
      the least recent trading day to specified end date.
"""
def load_prior_data(data_dir: Path,
    ticker: str, begin: str = "", end: str = ""
) -> pd.DataFrame | None:
    if not data_dir.is_dir():
        return None

    if not begin:
        start_date = min(
            date.fromisoformat(file.stem.removeprefix(f"{ticker}-"))
            for file in data_dir.iterdir()
            if (file.is_file())
        )
    else:
        start_date = date.fromisoformat(begin) if isinstance(begin, str) else begin
    if not end:
        end_date = max( date.fromisoformat(file.stem.removeprefix(f"{ticker}-"))
            for file in data_dir.iterdir()
            if (file.is_file())
        )
    else:
        end_date = date.fromisoformat(end) if isinstance(end, str) else end

    files: list[Path] = [
        f
        for f in data_dir.iterdir()
        if start_date
        <= date.fromisoformat(f.stem.removeprefix(f"{ticker}-"))
        <= end_date
    ]
    if not files:
        return None

    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


"""
Loads a tickers most recent trading day from disk.

@param ticker The ticker to use.
"""
def load_latest_day(data_dir: Path | str, ticker: str) -> pd.DataFrame | None:
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    if not data_dir.is_dir():
        print(f"No such directory for {ticker} exists.")
        return None

    target_file: Path | None = max(
        (file for file in data_dir.iterdir() if file.is_file()),
        key=lambda file: date.fromisoformat(file.stem.removeprefix(f"{ticker}-")),
        default=None,
    )
    if target_file is None:
        return None

    return pd.read_parquet(target_file)


"""
Loads a tickers most recent n observations from disk.

@param data_dir directory where files are located.
@param ticker The ticker to use.
@param n The number of observations to load from disk.
"""
def load_last_n(data_dir: Path, ticker: str, n: int) -> pd.DataFrame | None:
    if not data_dir.is_dir():
        return None

    files: list[Path] = [file for file in data_dir.iterdir() if file.is_file()]

    if not files:
        return None

    files.sort(
        key=lambda file: date.fromisoformat(file.stem.removeprefix(f"{ticker}-")),
        reverse=True,
    )

    dfs: list[pd.DataFrame] = []
    it = iter(files)
    while n > 0:
        next_file = next(it, None)
        if not next_file:
            break
        next_df: pd.DataFrame = (
            pd.read_parquet(next_file)
            .sort_values(by="window_start", ascending=False)
            .iloc[:n]
        )
        n -= next_df.shape[0]
        dfs.append(next_df)

    return pd.concat(dfs, ignore_index=True)


"""
Fetches OHLC data from massive.com for a specified lookback period.

@param period Lookback unit accepted by datettime's relativedelta such as 'days', 'weeks', 'months', 'years'.
@param depth The number of lookback units to fetch.
@param massive_parameters parameters to use in the call to the massive api.
@param client An instance of a massive Restclient.

@return A generator yielding OHLC data.
"""
def fetch_lookback(
    period: str, depth: int, massive_parameters: MassiveParameters, massive_client
) -> Iterator[FuturesAgg]:
    params = copy(massive_parameters)

    current_end_date: date = date.today() + timedelta(days=1)

    past_date: date = current_end_date - relativedelta(**{period: depth})  # type: ignore[arg-type]

    params["window_start_gte"] = past_date.isoformat()
    params["window_start_lte"] = current_end_date.isoformat()
    print(
        f"Fetching the last {depth} {period} of history for {massive_parameters['ticker']}."
    )
    return fetch_data(massive_client, params)


"""
Fetches OHLC from massive.com between a specifed date range.

@param begin The date to begin collection.
@param end The date to end collection.
@param massive_parameters parameters to use in the call to the massive api.
@param client An instance of a massive Restclient.

@return A generator yielding OHLC data.
"""
def fetch_range(
    begin: date, end: date, massive_parameters: MassiveParameters, massive_client
) -> Iterator[FuturesAgg]:

    params: MassiveParameters = copy(massive_parameters)
    params["window_start_gte"] = begin.isoformat()
    params["window_start_lte"] = (end + timedelta(days=1)).isoformat()

    print(f"Fetching data for {massive_parameters['ticker']} between {begin} and {end}")
    return fetch_data(massive_client, params)

"""
Fetches the latest (not present on disk) OHLC from massive.com.

@param massive_parameters parameters to use in the call to the massive api.
@param client An instance of a massive Restclient.

@return A generator yielding OHLC data.
"""
def fetch_latest(
    massive_parameters: MassiveParameters, massive_client
) -> Iterator[FuturesAgg | bytes]:
    params = copy(massive_parameters)
    ticker: str = massive_parameters["ticker"]
    data_dir = Path(RAW_DATA_DIR) / ticker

    # Load the latest observations from disk.
    last_observation: pd.DataFrame | None = load_last_n(data_dir, ticker, 1)

    if last_observation is None:
        print("No history for this ticker. Use --lookback or --range instead.")
        return

    # Get the observations next starting window from the most recent observation.
    latest_window_start: datetime = last_observation["window_start"].iloc[0]

    params["window_start_gte"] = latest_window_start.isoformat()

    print(
        f"Fetching the latest data for {massive_parameters['ticker']}"
        f"beginning at {latest_window_start}"
    )

    yield from fetch_data(massive_client, params)
