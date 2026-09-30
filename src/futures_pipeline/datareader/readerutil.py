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


"""
Returns the market open and close timestamps for each session in [begin, end].

@returns A dictionary object where each session date key contains a 
         two element list containing the begin and end of the session.

@note Fallback supports MES/ES equity-index futures schedules. 
"""
def fetch_session_dates(
    contract: str, 
    client: RESTClient, 
    begin: str, 
    end: str
) -> dict[str, list[pd.Timestamp]]:

    schedules = client.list_futures_schedules(
        product_code=contract,
        sort="product_code.asc",
        session_end_date_gte=begin,
        session_end_date_lte=end
    )
    df = pd.DataFrame.from_records(vars(d) for d in schedules)

    if not df.empty:
        for session, session_df in df.groupby('session_end_date'):
            if not {"open", "close"}.issubset(session_df['event']):
                df = df[df['session_end_date'] != session]

    missing_sessions = get_missing_sessions(
            df['session_end_date'].to_list() if not df.empty else [],
            begin,
            end
    )

    df = pd.concat([df, missing_sessions], ignore_index=True)
        
    if df.empty:
        raise ValueError('Failed to fetch schedule data.')

    df = df[df['event'].isin(['open','close'])]

    df['timestamp'] = pd.to_datetime(
            df['timestamp'],
            utc=True,
            format='mixed',
            errors='raise'
    ).dt.tz_convert('America/Chicago')

    df['session_end_date'] = pd.to_datetime(
            df['session_end_date'],
            errors='raise'
    ).dt.strftime("%Y-%m-%d")

    if df[['timestamp', 'session_end_date']].isna().any().any():
        raise ValueError("Schedule contains missing timestamps or session dates.")

    opens: pd.Series = (
        df[df["event"] == "open"].groupby("session_end_date")["timestamp"].min()
    )
    closes: pd.Series = (
        df[df["event"] == "close"].groupby("session_end_date")["timestamp"].max()
    )

    sessions = pd.concat([opens.rename("open"), closes.rename("close")], axis=1, join='outer')

    if sessions['open'].ge(sessions['close']).any():
        raise ValueError('A session must open before closing.')

    if any(sessions.isna().any()):
        raise ValueError("A session is missing its opening or closing timestamp.")

    sessions = sessions.apply(list, axis=1)

    return sessions.to_dict()

"""
Subroutine to fill in missing schedules since massive.com 
lacks schedule information for some dates. Uses pandas_market_calender 
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

    if not df.empty:
        for session, session_df in df.groupby('session_end_date'):
            if not {"open", "close"}.issubset(session_df['event']):
                df = df[df['session_end_date'] != session]

    # a session may have incomplete data, i.e a open but no close.
    missing_sessions = get_missing_sessions(
            df['session_end_date'].to_list() if not df.empty else [],
            begin,
            end
    )

    df = pd.concat([df, missing_sessions], ignore_index=True)

    df['timestamp'] = pd.to_datetime(
            df['timestamp'],
            utc=True,
            format='mixed',
            errors='raise'
    ).dt.tz_convert('America/Chicago')

    df['session_end_date'] = pd.to_datetime(df['session_end_date'], errors='raise').dt.strftime("%Y-%m-%d")

    if df[['timestamp', 'session_end_date']].isna().any().any():
        raise ValueError("Schedule contains missing timestamps or session dates.")

    # Remove the obsolete 15:15 equity-index pause.
    pre_opens = df[df['event'] == 'pre_open']['timestamp']
    maintenance = (pre_opens.dt.time == time(15,15))
    df = df.drop(pre_opens.index[maintenance]).reset_index(drop=True)

    df = df[['session_end_date', 'timestamp', 'event']]

    if df.empty:
        raise ValueError('Failed to fetch schedule data.')

    df = df.drop_duplicates(
            ['session_end_date', 'timestamp', 'event']
    )

    session_trading_hours = defaultdict(list)
    for end_date, d in df.groupby("session_end_date")[['timestamp', 'event']]:
        s: pd.Series = d.sort_values('timestamp').set_index('event')['timestamp']
        opened_at = None
        for event, timestamp in s.items():
            if event == 'open':
                if opened_at is not None:
                    raise ValueError(
                        f"Session {end_date} has consecutive openings: "
                        f"{opened_at} and {timestamp}."
                    )
                opened_at = timestamp
            elif event in ('pre_open', 'close') and opened_at is not None:
                if timestamp <= opened_at:
                    raise ValueError(f"Invalid trading interval for {end_date}.")
                session_trading_hours[end_date].append([opened_at, timestamp])
                opened_at = None

        if opened_at is not None:
            raise ValueError(f"Session {end_date} has an unmatched opening.")
    return session_trading_hours



def get_product_code(symbol: str):
    ticker_match = re.match(r"([A-Z0-9]+?)[FGHJKMNQUVXZ]\d{1,4}", symbol)
    if ticker_match:
        return ticker_match.group(1)

    product_match = re.match(r"[A-Z0-9]+", symbol)
    if product_match:
        return product_match.group(0)

    raise ValueError(f"Error parsing product code from {symbol}.")


def get_missing_sessions(non_missing: list[str], start_date: str, end_date: str):

    trade_dates = mcal.get_calendar("CME_TradeDate")

    expected = trade_dates.valid_days(
            start_date, end_date
    ).tz_localize(None).normalize()

    existing = pd.DatetimeIndex(
            pd.to_datetime(non_missing)
    ).normalize()

    missing = expected.difference(existing)

    missing_schedules = []
    for day in missing:

        schedule: pd.DataFrame = cme_session_dates_fallback(day.strftime("%Y-%m-%d"))
        if schedule.empty:
            raise ValueError(f"Missing schedule information for {day}.")

        schedule = schedule.rename(columns={"break_start" : "pre_open", 'market_open': 'open', "market_close" : "close"})
        schedule = schedule.reset_index(drop=True)
        schedule['session_end_date'] = day

        schedule = schedule.melt( 
            id_vars="session_end_date",
            value_vars=['open', 'close', 'pre_open'],
            var_name='event',
            value_name='timestamp'
        )

        missing_schedules.append(schedule)

    if not missing_schedules:
        return pd.DataFrame(
                columns=['session_end_date', 'event', 'timestamp']
        )

    return pd.concat(missing_schedules, ignore_index=True)
