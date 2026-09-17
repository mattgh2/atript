from datetime import timedelta, date, datetime, time
from zoneinfo import ZoneInfo
from massive.rest import RESTClient
import pandas as pd
from typing import Hashable


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
