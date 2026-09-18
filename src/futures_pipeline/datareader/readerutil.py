from datetime import timedelta, date, datetime, time
from zoneinfo import ZoneInfo
from collections import defaultdict
from massive.rest import RESTClient
import pandas as pd
from typing import Hashable
from ..config import load_settings
from ..massive_client import create_massive_client
import re

def session_bounds(first: date, last: date) -> tuple[datetime, datetime]:
    chicago = ZoneInfo("America/Chicago")

    start = datetime.combine(
        first - timedelta(days=1),
        time(17, 0),
        tzinfo=chicago,
    )
    end = datetime.combine(
        last,
        time(16, 0),
        tzinfo=chicago,
    )
    return start, end


def mes_roll_date(year: int, month: int) -> date:
    if month not in {3, 6, 9, 12}:
        raise ValueError("MES contract month must be 3, 6, 9, or 12")

    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)

    return third_friday - timedelta(days=4)


def fetch_session_dates(
    contract: str, client: RESTClient, begin: str, end: str
) -> dict[Hashable, list[pd.Timestamp]]:

    schedules = client.list_futures_schedules(
        product_code=contract,
        sort="product_code.asc",
        session_end_date_gte=begin,
        session_end_date_lte=end
    )

    df = pd.DataFrame.from_records(vars(d) for d in schedules)

    if df.empty:
        raise ValueError('Failed to fetch schedule data.')

    df = df[df['event'].isin(['open','close'])]

    df['timestamp'] = df['timestamp'].map(lambda t: pd.Timestamp(t, tz="America/Chicago"))

    opens: pd.Series = (
        df[df["event"] == "open"].groupby("session_end_date")["timestamp"].min()
    )
    closes: pd.Series = (
        df[df["event"] == "close"].groupby("session_end_date")["timestamp"].max()
    )
    sessions = pd.concat([opens.rename("open"), closes.rename("close")], axis=1).apply(
        list, axis=1
    )
    if sessions.isna().any():
        raise ValueError("A session is missing its opening or closing timestamp.")

    return sessions.to_dict()

def fetch_session_hours(product: str, begin: str, end: str, client: RESTClient | None = None) -> dict[str, list]:
    if client is None:
        client = create_massive_client(load_settings().massive_api_key)

    schedules = client.list_futures_schedules(
            product_code=product,
            session_end_date_gte=begin,
            session_end_date_lte=end
    )

    df = pd.DataFrame.from_records(vars(d) for d in schedules)
    if df.empty:
        raise ValueError('Failed to fetch schedule data.')

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors="coerce").dt.tz_convert("America/Chicago")

    session_trading_hours = defaultdict(list)
    for end_date, d in df.groupby("session_end_date")[['timestamp', 'event']]:
        s: pd.Series = d.sort_values('timestamp').set_index('event')['timestamp']
        opened_at = None
        for event, timestamp in s.items():
            if event == 'open':
                opened_at = timestamp
            elif event in ('pre_open', 'close') and opened_at is not None:
                session_trading_hours[end_date].append([opened_at, timestamp])
                opened_at = None

    return session_trading_hours


def get_product_code(symbol: str):
    ticker_match = re.match(r"([A-Z0-9]+?)[FGHJKMNQUVXZ]\d{1,4}", symbol)
    if ticker_match:
        return ticker_match.group(1)

    product_match = re.match(r"[A-Z0-9]+", symbol)
    if product_match:
        return product_match.group(0)

    raise ValueError(f"Error parsing product code from {symbol}.")
