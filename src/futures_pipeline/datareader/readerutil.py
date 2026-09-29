from datetime import timedelta, date, datetime, time
from zoneinfo import ZoneInfo
from collections import defaultdict
from massive.rest import RESTClient
import pandas as pd
from typing import Hashable
from ..config import load_settings
from ..massive_client import create_massive_client
import re
from tqdm import tqdm
import pandas_market_calendars as mcal
from logging import Logger

logger = Logger(name='log', level=2)

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


# NOTE: MUST BE CALLED WITH A CONTRACT FROM THE CME.
def fetch_session_dates(
    contract: str, 
    client: RESTClient, 
    begin: str, 
    end: str
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
    df['session_end_date'] = pd.to_datetime(
            df['session_end_date']
    )

    opens: pd.Series = (
        df[df["event"] == "open"].groupby("session_end_date")["timestamp"].min()
    )
    closes: pd.Series = (
        df[df["event"] == "close"].groupby("session_end_date")["timestamp"].max()
    )

    sessions = pd.concat([opens.rename("open"), closes.rename("close")], axis=1, join='inner')

    if any(sessions.isna().any()):
        raise ValueError("A session is missing its opening or closing timestamp.")

    sessions = sessions.apply(list, axis=1)

    min_session = df['session_end_date'].min()
    max_session = df['session_end_date'].max()

    num_days = (max_session - min_session).days
    session_index = set(sessions.index)

    trade_dates = mcal.get_calendar("CME_TradeDate")

    for day in tqdm(
        (min_session + timedelta(days=offset) for offset in range(1, num_days + 1)), 
        desc="Filling in missing sessions using pandas_market_calendars api"
    ):
        if day not in session_index:
            if trade_dates.valid_days(day, day).empty or day in session_index:
                continue

            schedule: pd.DataFrame = cme_session_dates_fallback(day)
            open = pd.Timestamp(schedule.iloc[0]['market_open']), 
            close = pd.Timestamp(schedule.iloc[-1]['market_close'])
            sessions.loc['session_end_date'] = [open, close]

    return sessions.to_dict()

"""
Subroutine for ``fetch_session_dates`` if massive.com 
lacks schedule information for a specified date, in which case the 
pandas_market_calender api shall be called.
"""
def cme_session_dates_fallback(
    trade_date: str,
) -> pd.DataFrame:

    equity = mcal.get_calendar("CME_Equity")
    trade_dates = mcal.get_calendar("CME_TradeDate")
    target = pd.Timestamp(trade_date).normalize()
    day = pd.Timedelta(days=1)

    if trade_dates.valid_days(target, target).empty:
        raise ValueError(f"{trade_date} is not a valid trading date.")

    # Locate the preceding business trade date.
    previous = target - day
    while trade_dates.valid_days(previous, previous).empty:
        previous -= day

    # Include schedule days after the preceding trade date.
    schedule = equity.schedule(
        start_date=previous + day,
        end_date=target,
        tz="America/Chicago",
    )


    return schedule

def fetch_session_hours(product: str, begin: str, end: str, client: RESTClient | None = None) -> dict[str, list]:
    if client is None:
        client = create_massive_client(load_settings().massive_api_key)

    schedules = client.list_futures_schedules(
            product_code=product,
            session_end_date_gte=begin,
            session_end_date_lte=end
    ) 

    df = pd.DataFrame.from_records(vars(d) for d in schedules)

    start_session = pd.Timestamp(begin)
    end_session = pd.Timestamp(end)

    num_days = (end_session - start_session).days
    sessions = set(df['session_end_date'] if not df.empty else ())
    trade_dates = mcal.get_calendar("CME_TradeDate")

    missing_schedules = []
    for day in (start_session + timedelta(days=offset) for offset in range(num_days + 1)):
        if trade_dates.valid_days(day,day).empty or day in sessions:
            continue

        schedule: pd.DataFrame = cme_session_dates_fallback(day.isoformat())
        schedule = schedule.rename(columns={"break_start" : "pre_open", 'market_open': 'open', "market_close" : "close"})
        schedule = schedule.reset_index(names=['session_end_date'])

        schedule = schedule.melt( 
            id_vars="session_end_date",
            value_vars=['open', 'close', 'pre_open'],
            var_name='event',
            value_name='timestamp'
        )

        # Filter out maintenance breaks
        pre_opens = schedule[schedule['event'] == 'pre_open']['timestamp']
        maintenance = (pre_opens.dt.time == time(15,15))
        schedule = schedule.drop(pre_opens.index[maintenance]).reset_index(drop=True)
        missing_schedules.append(schedule)

    df = pd.concat([df, *missing_schedules])
    df = df[['session_end_date', 'timestamp', 'event']]

    if df.empty:
        raise ValueError('Failed to fetch schedule data.')

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors="coerce").dt.tz_convert("America/Chicago")
    df['session_end_date'] = pd.to_datetime(df['session_end_date'])


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
