from dataclasses import dataclass
from typing import Required, TypedDict
from pandas import Series, DataFrame
from enum import IntEnum
from .validate import (
    CommandArgs,
    FetchLatestArgs,
    FetchRangeArgs,
    FetchLookbackArgs,
    ModelArgs,
    PreprocessArgs,
    TradingFees,
    TradingFeeUpdates,
    ContractSpec,
    PredictionInterval
)
from typing import Literal
import re

@dataclass
class Limit:
    limit: int

class MassiveParameters(TypedDict, total=False):
    ticker: Required[str]
    limit: int
    sort: str
    resolution: Required[str]
    window_start:  str
    window_start_gte: str
    window_start_lte: str
    window_start_gt: str
    window_start_lt: str


@dataclass
class EvaluationResult:
    pred_df: Series
    eval_df: Series 
    by_horizon: DataFrame
    mae: float
    baseline_mae: float
    mae_skill: float
    rmse: float
    directional_accuracy: float
    by_horizon_loss: DataFrame
    loss: dict[float, float]
    baseline_loss: dict[float,float]
    intervals_by_horizon: DataFrame
    pi_coverage: float
    nominal_coverage: float
    mean_interval_width: float
    quantile_calibration: dict[float, float]
    calibration_error: dict[float, float]

@dataclass
class BacktestResults:
    trade_count: int
    net_pnl: float
    win_rate: float
    average_pnl: float
    gross_profit: float
    gross_loss: float

    def display_results(self) -> None:
        print(f"Trade count:   {self.trade_count:,}")
        print(f"Net P&L:       ${self.net_pnl:,.2f}")
        print(f"Win rate:      {self.win_rate:.2%}")
        print(f"Average P&L:   ${self.average_pnl:,.2f}")
        print(f"Gross profit:  ${self.gross_profit:,.2f}")
        print(f"Gross loss:    ${self.gross_loss:,.2f}")



type InputArgs = (
        CommandArgs 
        | FetchLookbackArgs 
        | FetchRangeArgs 
        | FetchLatestArgs 
        | ModelArgs 
        | PreprocessArgs
        | TradingFeeUpdates
)

type FetchArgs = (
        CommandArgs
        | FetchLookbackArgs 
        | FetchRangeArgs 
        | FetchLatestArgs 
)

type TimedeltaUnit = Literal["s", "min", "h", "D", "W"]

class CandleResolution:
    resolution: str
    length: int
    unit: str

    __fixed_units: dict[str, TimedeltaUnit] = {
            "sec": "s",
            "min": "min",
            "hour": 'h',
            "day": "D",
            "week": "W"
    }

    def __init__(self, resolution: str) -> None:
        self.resolution = resolution
        #  Parse resolution.
        match = re.match(r"^(\d+)(sec|min|day|hour|session|week|month|quarter|year)$", resolution)
    
        if (match is None):
            raise ValueError(f"{resolution} is not a valid resolution.")
    
        self.length = int(match.group(1))
        self.unit = match.group(2)

    # Maps resolution units to a unit from TimeDeltaUnitChoices.
    def to_timedelta_unit(self) -> TimedeltaUnit:
        return self.__fixed_units[self.unit]


class Signal(IntEnum):
    LONG = 1
    SHORT = -1
    HOLD = 0
