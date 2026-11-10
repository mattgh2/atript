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
    PredictionInterval,
    FetchTrainArgs
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
    y_pred: Series
    y_true: Series 
    lower_p: Series
    upper_p: Series
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

    def to_strings(self) -> list[str]:
        """Format the evaluation metrics for terminal or HTML reporting."""
        output = [
            "=== POINT FORECAST METRICS ===",
            "=== BY HORIZON ===",
            *(
                f"H{h}: count={row['count']:.0f}, "
                f"MAE={row['mae']:.4f}, "
                f"RMSE={row['rmse']:.4f}, "
                f"direction={row['directional_accuracy']:.2%}"
                for h, row in self.by_horizon.iterrows()
            ),
            f"MAE: {self.mae:.8f}",
            f"Baseline MAE: {self.baseline_mae:.8f}",
            f"MAE Skill: {self.mae_skill:.2%}",
            f"RMSE: {self.rmse:.8f}",
            f"Directional accuracy: {self.directional_accuracy:.8f}",
            "=== QUANTILE LOSS ===",
            *(

                f"H{h}: " + ", ".join(f"q={q}: {value:.6f}" for q, value in row.items())
                for h, row in self.by_horizon_loss.iterrows()
            ),
        ]

        output.extend(
            f"q={quantile:.2%}: {loss:.10f}"
            for quantile, loss in sorted(self.loss.items())
        )

        if self.baseline_loss:
            output.append("=== BASELINE QUANTILE LOSS ===")
            output.extend(
                f"q={quantile:.2%}: {loss:.10f}"
                for quantile, loss in sorted(self.baseline_loss.items())
            )

        output.append("=== QUANTILE CALIBRATION ===")
        output.extend(
            f"q={quantile:.2%}: {calibration:.2%}"
            for quantile, calibration in sorted(self.quantile_calibration.items())
        )

        output.append("=== CALIBRATION ERROR ===")
        output.extend(
            f"q={quantile:.2%}: {error:+.2%}"
            for quantile, error in sorted(self.calibration_error.items())
        )

        output.extend([
            "=== PREDICTION INTERVAL ===",
            *(
                f"H{h}: coverage={row['coverage']:.2%}, width={row['mean_width']:.4f}"
                for h, row in self.intervals_by_horizon.iterrows()
            ),
            f"PI coverage: {self.pi_coverage:.2%}",
            f"Nominal coverage: {self.nominal_coverage:.2%}",
            f"Mean PI width: {self.mean_interval_width:.8f}",
        ])

        return output

@dataclass
class BacktestResults:
    trade_count: int
    net_pnl: float
    win_rate: float
    average_pnl: float
    gross_profit: float
    gross_loss: float
    missing_count: int
    net_sharpe: float

    def to_strings(self) -> list[str]:
        """Format backtest metrics for terminal or HTML reporting."""
        return [
            "=== BACKTEST RESULTS ===",
            f"Trade count: {self.trade_count:,}",
            f"Net P&L: ${self.net_pnl:,.2f}",
            f"Win rate: {self.win_rate:.2%}",
            f"Average P&L: ${self.average_pnl:,.2f}",
            f"Gross profit: ${self.gross_profit:,.2f}",
            f"Gross loss: ${self.gross_loss:,.2f}",
            f"Total missing rows: {self.missing_count}",
            f"Net Sharpe per window: {self.net_sharpe:.2f}",
        ]

    def display_results(self) -> None:
        print("\n".join(self.to_strings()))



type InputArgs = (
        CommandArgs 
        | FetchLookbackArgs 
        | FetchRangeArgs 
        | FetchLatestArgs 
        | ModelArgs 
        | PreprocessArgs
        | TradingFeeUpdates
)

type FetchContextArgs = (
        FetchLookbackArgs 
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
