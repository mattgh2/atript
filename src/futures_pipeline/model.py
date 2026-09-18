import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from pathlib import Path
from chronos import Chronos2Pipeline
from .config import PROCESSED_DATA_DIR
from .typedefs import EvaluationResult, TradingFees, ContractSpec, Signal, PredictionInterval, BacktestResults
from .datareader import load_prior_data, fetch_contract_spec
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import load_fees

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def fine_tuned_model(ticker, target, pred_length, quantiles):

    volatility = ['rolling_std_20', "true_range"]
    trend = ["close_to_ema", "ema_slope"]
    momentum = ["rsi"]
    relative_position = ['percent_b', 'close_to_vwap']
    market_activity = ['volume']
    time = ['has_time_gap', 'log_elapsed_intervals']

    covariates: list = [*volatility, *trend, *momentum, *relative_position, *market_activity, *time]

    predictor = TimeSeriesPredictor(
            prediction_length=pred_length,
            target=target,
            known_covariates_names=covariates,
            eval_metric="MASE"
    ).fit(
            # train_data  = []
    )

def run_model(
    ticker,
    target,
    pred_length,
    quantiles,
    prediction_interval: PredictionInterval,
    contract_spec: ContractSpec,
    hf_token=None,
    model_dir=None,
    store_weights: bool = False,
    eval: bool=False
) -> None:
    pipeline = load_chronos(model_dir, store_weights=store_weights, hf_token=hf_token)

    # Load historical target values and past values of covariates
    data: pd.DataFrame | None = load_prior_data(PROCESSED_DATA_DIR / ticker, ticker)

    if data is None:
        raise RuntimeError("No data.")

    volatility = ['rolling_std_20', "true_range"]
    trend = ["close_to_ema", "ema_slope"]
    momentum = ["rsi"]
    relative_position = ['percent_b', 'close_to_vwap']
    market_activity = ['volume']
    time = ['has_time_gap', 'log_elapsed_intervals']

    # covariates: list = [*volatility, *trend, *momentum, *relative_position, *market_activity, *time]
    covariates: list = []

    context_df = (
        data[["model_timestamp", "ticker", target, *covariates]]
        .sort_values("model_timestamp")
        .reset_index(drop=True)
    )

    if eval:
        initial_train_size = int(context_df.shape[0] * .80)
        e = walk_forward_predict(pipeline, context_df, pred_length, pred_length,initial_train_size, target, quantiles)
        eval_results: EvaluationResult = evaluate(e, target, quantiles, prediction_interval)

        print(f"=== POINT FORECAST METRICS ===")
        print(eval_results.by_horizon.to_string())
        print(f"MAE: {eval_results.mae:.8f}")
        print(f"Baseline MAE: {eval_results.baseline_mae:.8f}")
        print(f"MAE Skill: {eval_results.mae_skill:.2%}")
        print(f"RMSE: {eval_results.rmse:.8f}")
        print(f"Directional accuracy: {eval_results.directional_accuracy:.8f}\n")

        loss_skill = {
                q: 1 - eval_results.loss[q] / eval_results.baseline_loss[q]
                if eval_results.baseline_loss[q] != 0 else np.nan
                for q in quantiles
        }

        print(f"=== QUANTILE LOSS ===")
        print(eval_results.by_horizon_loss.to_string())
        for t in zip(
            eval_results.loss.items(),
            eval_results.baseline_loss.items(),
            loss_skill.items(),
        ):
            (quantile, avg_loss), (_, baseline_loss), (_, loss_skill) = t
            print(f"avg loss for {quantile:.2%}th quantile: {avg_loss:.10f}")
            print(f"avg baseline loss for {quantile:.2%}th quantile: {baseline_loss:.10f}")
            print(f"loss skill for {quantile:.2%}th quantile: {loss_skill:.2%}\n")

        print(f" === Quantile Calibration === ")
        for q, v in eval_results.quantile_calibration.items():
            print(f"q={q}: {v}")

        print(f" \n=== Calibration Error === ")
        for q, v in eval_results.calibration_error.items():
            print(f"q={q}: {v}")

        print(f"\n=== PREDICTION INTERVAL ===")
        print(eval_results.intervals_by_horizon.to_string())
        print(f"Mean PI coverage: {eval_results.pi_coverage:.2%}")
        print(f"Nominal coverage: {eval_results.nominal_coverage:.2%}")
        print(f"Mean PI width: {eval_results.mean_interval_width:.8f}")

    else:
        print(f"Performing walk-forward forecasting")
        forecasts = walk_forward_predict(
            pipeline,
            context_df,
            1,
            1,
            int(context_df.shape[0] * 0.98),
            target,
            quantiles,
        )

        market_data = data[['ticker', 'open', 'close', 'model_timestamp']].copy()
        print("Running backtesting...")
        backtest_results: BacktestResults = backtest_strategy(forecasts, market_data, prediction_interval, contract_spec)
        backtest_results.display_results()

        # pred_df = predict_chronos(pipeline, context_df, pred_length, target, quantiles)
        #
        # ts_context = context_df.set_index('model_timestamp')[target].tail(256)
        # ts_pred = pred_df.set_index("model_timestamp")
        #
        # ts_context.plot(label="historical data", figsize=(12,3))
        # ts_pred['predictions'].plot(label="forecast")
        #
        # plt.fill_between(
        #         ts_pred.index,
        #         ts_pred[str(prediction_interval[0])],
        #         ts_pred[str(prediction_interval[1])],
        #         alpha=0.7,
        #         label="prediction interval"
        # )
        # plt.legend()
        # plt.show()


def predict_chronos(
    pipeline: Chronos2Pipeline,
    context_df: pd.DataFrame,
    pred_length: int,
    target: str,
    quantiles,
) -> pd.DataFrame:

    # Generate predictions with covariates
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=pred_length,  # Number of steps to forecast
        quantile_levels=quantiles,  # Quantile for probabilistic forecast
        id_column="ticker",  # Column identifying different time series
        timestamp_column="model_timestamp",  # Column with datetime information
        target=target,  # Column(s) with time series values to predict
    )
    return pred_df


"""
Loads a Chronos model from local storage or Hugging Face.

@param model_dir the name of the directory where the model is located.
@param store_weights Boolean flag for storing weights locally.
@param hf_token A Hugging Face token.

@returns A ``Chronos2Pipeline``
"""
def load_chronos(model_dir: Path | None = None, store_weights: bool=False, hf_token: str | None=None) -> Chronos2Pipeline:
    pipeline: Chronos2Pipeline | None = None

    if model_dir is not None and model_dir.exists():
        try:
            pipeline = Chronos2Pipeline.from_pretrained(
                model_dir, device_map="cuda", local_files_only=True
            )
        except (OSError, ValueError) as e:
            print(f"Error occured while loading chronos-2 from {model_dir}: {str(e)}")

    if pipeline is None:
        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map="cuda", token=hf_token
        )

    if store_weights:
        assert model_dir is not None, "model_dir is required when store_weights=True"
        pipeline.save_pretrained(model_dir)

    return pipeline

"""
Computes the element-wise pinball loss for a quantile forecast.

Pinball loss is asymmetric. Underpredictions receive weight ``quantile``,
while overpredictions receive weight ``1 - quantile``. Higher quantiles
therefore penalize underprediction more heavily.

The loss for quantile ``q`` is:
    q * (y_true - y_pred)          if y_true >= y_pred
    (q - 1) * (y_true - y_pred)    otherwise

@param y_true Observed target values.
@param y_pred Predicted values for the specified quantile.
@param quantile Quantile level associated with ``y_pred``

@returns: A NumPy array containing one nonnegative loss value per observation.
"""
def pinball_loss(y_true, y_pred, quantile) -> np.ndarray:
    return np.where(
        y_true >= y_pred,
        quantile * (y_true - y_pred),
        (quantile - 1) * (y_true - y_pred),
    )


"""
Performs walk-forward evaluation using an expanding window to obtain a set of evaluation data points.

@param pipline A chronos2 pipeline instance.
@param context_df the set of observations fed to the model.
@param step The number of new observations to include in each iteration.
@param horizon The amount of values to forecast.
@param initial_train_size The size of context_df excluding its hold out set.
@param target The models target feature.
@param quantiles the quantiles used in forecasting.

@Note: For a fair baseline quantile loss, a new baseline value must be computed for every expansion.
"""
def walk_forward_predict(
    pipeline: Chronos2Pipeline,
    context_df: pd.DataFrame,
    step: int,
    horizon: int,
    initial_train_size: int,
    target: str,
    quantiles: list[float],
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    last = context_df.shape[0] - horizon
    for window_len in range(initial_train_size, last + 1, step):
        print(f"{window_len}/{last}")
        pred_df = predict_chronos(pipeline, context_df[: window_len], horizon, target, quantiles)
        pred_df["horizon"] = np.arange(1, len(pred_df) + 1)
        pred_df['forecast_origin'] = context_df.iloc[window_len - 1]['model_timestamp']

        # Compute each quantiles baseline value for the current context window. (historical quantile)
        for quantile in quantiles:
            if np.isclose(quantile, 0.5):
                baseline_value = 0.0
            else:
                baseline_value = context_df[target].iloc[:window_len].quantile(quantile)
            pred_df[f"baseline_{quantile:g}"] = baseline_value

        eval_df = context_df[['ticker','model_timestamp', target]].iloc[window_len: window_len + horizon]
        results.append(pred_df.merge(
                eval_df,
                on=['ticker','model_timestamp'],
                how="inner",
                validate="one_to_one"
        ))
    df = pd.concat(results, ignore_index=True)
    return df

"""
Computes point, quantile, interval, calibration, and horizon-based
forecast metrics for an evaluation set.

Metrics:
    - Mean absolute error (MAE)
    - Root mean squared error (RMSE)
    - Directional accuracy
    - MAE relative to a zero-return baseline
    - MAE skill score
    - Pinball loss for each quantile
    - Quantile baseline loss and skill score
    - Prediction-interval coverage
    - Mean prediction-interval width
    - Quantile calibration and calibration error
    - Point and probabilistic metrics grouped by forecast horizon
    
@param eval  Evaluation observations and forecasts.
@param target Name of the column containing the observed target values.
@param prediction_interval
@param quantiles Quantile levels to evaluate.
@param prediction_interval  Lower and upper quantile levels defining the prediction interval.

@returns  An ``EvaluationResult`` containing the evaluation metrics.
"""
def evaluate(
        eval: pd.DataFrame,  target: str, quantiles: list[float], prediction_interval: PredictionInterval
) -> EvaluationResult:

    actual = eval[target]
    predicted = eval["predictions"]
    error = actual - predicted

    by_horizon_pf = eval.groupby("horizon").apply(
            lambda group: pd.Series({
                "count": len(group),
                "mae": ( group[target] - group['predictions']
                ).abs().mean(),
                "rmse": np.sqrt(
                    (group[target] - 
                     group['predictions']
                    ).pow(2).mean()
                ),
                "directional_accuracy": (
                    np.sign(group[target]) == np.sign(group["predictions"])
                ).mean()
                })
            , include_groups=False) # type: ignore

    mae: float = error.abs().mean()
    rmse: float = error.pow(2).mean() ** 0.5

    directional_accuracy = (np.sign(predicted) == np.sign(actual)).mean()

    # Predicting every future return is zero as a baseline for mae.
    baseline_mae = actual.abs().mean()

    # Measures how much the model improves upon the baseline.
    mae_skill = 1 - mae / baseline_mae

    by_horizon_loss: dict[str, pd.Series] = {}
    for q in quantiles:
        row_loss = pd.Series(
            pinball_loss(eval[target], eval[str(q)], q)
        , index=eval.index, dtype=float)
        row_baseline_loss = pd.Series(
                pinball_loss(eval[target], eval[f"baseline_{q:g}"], q)
        , index=eval.index, dtype=float)

        by_horizon_loss[str(q)] = row_loss.groupby(eval['horizon']).mean()
        by_horizon_loss[f"baseline_{q:g}"] = row_baseline_loss.groupby(eval['horizon']).mean()
        by_horizon_loss[f"{q:g}_skill"]= 1 - (by_horizon_loss[str(q)] / by_horizon_loss[f"baseline_{q:g}"].replace(0, np.nan))

    by_horizon_loss_df = pd.DataFrame(by_horizon_loss)

    # Overall quantile loss.
    loss: dict[float, float] = {}
    baseline_loss: dict[float, float] = {}
    for quantile in quantiles:
        loss[quantile] = pinball_loss(eval[target], eval[str(quantile)], quantile).mean()
        baseline_loss[quantile] = pinball_loss(actual, eval[f'baseline_{quantile:g}'], quantile).mean()

    def horizon_interval_metrics(group):
        y_true = group[target]
        lower = group[str(prediction_interval.lower)]
        upper = group[str(prediction_interval.upper)]

        return pd.Series({
            "coverage": ((lower <= y_true) & (y_true <= upper)).mean(),
            "mean_width": (upper - lower).mean()
        })


    intervals_by_horizon: pd.DataFrame = (
            eval.groupby("horizon")
            .apply(horizon_interval_metrics, include_groups=False) # type: ignore
    )

    mean_pi_coverage: float = (
        (eval[str(prediction_interval.lower)] <= eval[target])
        & (eval[target] <= eval[str(prediction_interval.upper)])
    ).mean()

    nominal_coverage = prediction_interval.upper - prediction_interval.lower
    mean_interval_width = (eval[str(prediction_interval.upper)] - eval[str(prediction_interval.lower)]).mean()

    # Ratio of the amount of actual values below its prediction.
    quantile_calibration: dict[float, float] = {
            q: (actual <= eval[str(q)]).mean()
            for q in quantiles
    }
    calibration_error: dict[float, float] = {
            q: quantile_calibration[q] - q
            for q in quantiles
    }

    ts_eval = eval.set_index("model_timestamp")
    return EvaluationResult (
        ts_eval['predictions'], 
        ts_eval[target], 
        by_horizon_pf,
        mae, 
        baseline_mae, 
        mae_skill, 
        rmse, 
        directional_accuracy, 
        by_horizon_loss_df,
        loss, 
        baseline_loss,
        intervals_by_horizon,
        mean_pi_coverage,
        nominal_coverage,
        mean_interval_width,
        quantile_calibration,
        calibration_error,
    )


# NOTE: Hardcoded to returns for now.
"""
Performs Backtest model evaluation on a set of forecasts.

Metrics:
    - Trade Count
    - Net pnl
    - Win rate
    - Average pnl
    - Gross profits
    - Gross loss

@param
@param
@param
@param
@param
"""
def backtest_strategy(
    forecasts: pd.DataFrame,
    market_data: pd.DataFrame,
    pred_interval: PredictionInterval,
    contract_spec: ContractSpec,
) -> BacktestResults:
    fees: TradingFees = load_fees()

    round_trip_cost: float = (
        fees.entry_fee
        + fees.exit_fee
        + fees.entry_commission
        + fees.exit_commission
    )

    execution_cost = (
            fees.spread + 2 * fees.slippage
    ) * contract_spec.price_per_tick

    total_round_trip_cost: float = round_trip_cost + execution_cost

    print(total_round_trip_cost)
    origin_prices = market_data[["ticker", "model_timestamp", "close"]].rename(
            columns={"model_timestamp": "forecast_origin", "close": "origin_close"}
    )

    backtest: pd.DataFrame = forecasts.merge(
            origin_prices,
            on=['ticker', 'forecast_origin'],
            how='left',
            validate='many_to_one'
    )

    backtest['threshold'] = pd.Series(
            calculate_cost_threshold(price, total_round_trip_cost, contract_spec.multiplier) for price in backtest['origin_close']
    )


    backtest['signal'] = pd.Series(
            # calcuate_signal_pi(row['threshold'], row[str(pred_interval.lower)], row[str(pred_interval.upper)])
            calculate_signal_point(row['threshold'], row['predictions'])
            for _, row in backtest.iterrows()
    )
    print(backtest['signal'].value_counts())
    for idx, row in backtest.iterrows():
        # print(f"Threshold: {row['threshold']:.8f}. PI: [{row[str(pred_interval.lower)]}, {row[str(pred_interval.upper)]}]. Signal is {row['signal']}")
        print(f"Threshold: {row['threshold']:.8f}. Return prediction: {row['predictions']:.8f}. Actual target value: {row['returns']:.8f}. AE: {abs(row['returns'] - row['predictions'])}. Signal is {row['signal']}")


    backtest = backtest.sort_values(["ticker", "forecast_origin"]).reset_index(
        drop=True
    )
    outcomes = market_data[['ticker', 'model_timestamp', 'open', 'close']].rename(
            columns={"open": "entry_price", "close": "exit_price"}
    )
    backtest = backtest.merge(
            outcomes,
            on=['ticker', 'model_timestamp'],
            how="left",
            validate="one_to_one"
    )
    backtest["gross_pnl"] = (
        backtest["signal"]
        * (backtest["exit_price"] - backtest["entry_price"])
        * contract_spec.multiplier
    )

    backtest['trading_cost'] = np.where(
            backtest['signal'] != Signal.HOLD,
            total_round_trip_cost,
            0.0
    )

    backtest['net_pnl'] = backtest['gross_pnl'] - backtest['trading_cost']

    trades = backtest[backtest['signal'] != Signal.HOLD]
    trade_count = trades.shape[0]
    net_pnl = trades['net_pnl'].sum()
    win_rate = (trades['net_pnl'] > 0).mean()
    average_pnl = trades['net_pnl'].mean()

    gross_profit = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = -trades[trades['net_pnl'] < 0]['net_pnl'].sum()

    return BacktestResults(
        trade_count, net_pnl, win_rate, average_pnl, gross_profit, gross_loss
    )


"""
Calculates a cost threshold expressed as a proportion of the
current notitional value.
"""
def calculate_cost_threshold(
    price: float, total_round_trip_cost: float, multiplier: float
) -> float:
    return total_round_trip_cost / (price * multiplier)


def calcuate_signal_pi(threshold, lower_b, upper_b) -> Signal:
    if lower_b > threshold:
        signal = Signal.LONG
    elif upper_b < -threshold:
        signal = Signal.SHORT
    else:
        signal = Signal.HOLD
    return signal

def calculate_signal_point(threshold, forecast):
    if forecast > threshold:
        signal = Signal.LONG
    elif forecast < -threshold:
        signal = Signal.SHORT
    else:
        signal = Signal.HOLD
    return signal


