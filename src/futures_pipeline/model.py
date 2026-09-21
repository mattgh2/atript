import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from pathlib import Path
from chronos import Chronos2Pipeline
from .config import PROCESSED_DATA_DIR, TRAINING_DATA_DIR
from .typedefs import EvaluationResult, TradingFees, ContractSpec, Signal, PredictionInterval, BacktestResults
from .datareader import load_prior_data, fetch_contract_spec
from .datareader.readerutil import get_product_code
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import load_fees
import torch

dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
)

TREND = ['EMA', 'SMA', 'DMA', 'T3MA']
MOMENTUM = ["RSI", 'FSO', 'SSO', 'ROC', 'CCI']
VOLATILITY = ["VR", "ATR", "percent_b"]
VOLUME = ["volume", "VWAP"]
TIME = ['has_time_gap', 'log_elapsed_intervals']

def build_training_input(train_df, covariates: list[str], pred_length) -> list:
    train = []
    for ticker, contract_df in train_df.groupby('ticker', sort=False):

        contract_df = contract_df.sort_values("real_timestamp").reset_index(drop=True).dropna(
            subset=['close']
        )

        if contract_df.shape[0] < 2  * pred_length:
            continue

        past_covariates = {
            column: np.array(values)
            for column, values in contract_df[[*covariates]].to_dict(orient="list").items()
        }

        train.append(
            {
                "target": contract_df['close'].to_numpy(),
                "past_covariates": past_covariates,
                # "future_covariates": {} # TODO: load future time covariates
            }
        ) 
    if not train:
        raise ValueError("No contracts contain enough training observations.")

    return train

def run_model(
    ticker,
    pred_length,
    quantiles,
    prediction_interval: PredictionInterval,
    contract_spec: ContractSpec,
    hf_token=None,
    model_dir=None,
    store_weights: bool = False,
    eval: bool=False
) -> None:

    product_code = get_product_code(ticker)

    history_df: pd.DataFrame | None = load_prior_data(
        PROCESSED_DATA_DIR / ticker, ticker
    )

    train_df: pd.DataFrame | None = load_prior_data(
        PROCESSED_DATA_DIR / product_code, product_code
    )

    if history_df is None:
        raise RuntimeError("Missing context set.")
    if train_df is None:
        raise RuntimeError("Missing training set.")

    train_df = train_df.sort_values(['ticker','real_timestamp'])

    covariates: list = [*VOLATILITY, *TREND, *MOMENTUM, *VOLUME, *TIME]

    context_df = (
        history_df[["model_timestamp", "ticker", "close", *covariates]]
        .sort_values("model_timestamp")
        .reset_index(drop=True)
    )

    train = build_training_input(train_df, covariates, pred_length)
    pipeline = load_chronos(train, model_dir, store_weights=store_weights, hf_token=hf_token, pred_length=pred_length)

    if eval:
        initial_train_size = int(context_df.shape[0] * .80)
        e = walk_forward_predict(pipeline, context_df, pred_length, pred_length,initial_train_size, quantiles)
        eval_results: EvaluationResult = evaluate(e, quantiles, prediction_interval)

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
            quantiles,
        )

        market_data = history_df[['ticker', 'open', 'close', 'model_timestamp']].copy()
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
    quantiles,
) -> pd.DataFrame:

    # Generate predictions with covariates
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=pred_length,  # Number of steps to forecast
        quantile_levels=quantiles,  # Quantile for probabilistic forecast
        id_column="ticker",  # Column identifying different time series
        timestamp_column="model_timestamp",  # Column with datetime information
        target="close",  # Column(s) with time series values to predict
    )
    return pred_df


"""
Loads a Chronos model from local storage or Hugging Face.

@param model_dir the name of the directory where the model is located.
@param store_weights Boolean flag for storing weights locally.
@param hf_token A Hugging Face token.

@returns A ``Chronos2Pipeline``
"""
def load_chronos(
    train,
    model_dir: Path | None = None,
    store_weights: bool = False,
    hf_token: str | None = None,
    pred_length: int = 24,
) -> Chronos2Pipeline:
    pipeline: Chronos2Pipeline | None = None

    # if model_dir is not None and model_dir.exists():
    #     try:
    #         pipeline = Chronos2Pipeline.from_pretrained(
    #             model_dir, device_map="cuda", local_files_only=True
    #         )
    #     except (OSError, ValueError) as e:
    #         print(f"Error occured while loading chronos-2 from {model_dir}: {str(e)}")

    # if pipeline is None:
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda", token=hf_token
    ).fit(
        inputs=train,
        prediction_length=pred_length,
        finetune_mode="full",
        learning_rate=1e-6,
        num_steps=1000,
        batch_size=64,
        context_length=256,
    )

    # if store_weights:
    #     assert model_dir is not None, "model_dir is required when store_weights=True"
    #     pipeline.save_pretrained(model_dir)

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
    quantiles: list[float],
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    last = context_df.shape[0] - horizon
    for window_len in range(initial_train_size, last + 1, step):
        print(f"{window_len}/{last}")
        pred_df = predict_chronos(pipeline, context_df[: window_len], horizon, quantiles)
        pred_df["horizon"] = np.arange(1, len(pred_df) + 1)
        pred_df['forecast_origin'] = context_df.iloc[window_len - 1]['model_timestamp']
        pred_df['origin_close'] = context_df.iloc[window_len - 1]['close']

        # Compute each quantiles baseline value for the current context window. (historical quantile)
        for quantile in quantiles:
            baseline_value = pred_df['origin_close']
            pred_df[f"baseline_{quantile:g}"] = baseline_value

        eval_df = context_df[['ticker','model_timestamp', 'close']].iloc[window_len: window_len + horizon]
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
        eval: pd.DataFrame, quantiles: list[float], prediction_interval: PredictionInterval
) -> EvaluationResult:

    actual: pd.Series = eval['close']
    predicted: pd.Series = eval["predictions"]
    origin_close: pd.Series = eval['origin_close']
    error = predicted - actual 
    
    def directional_accuracy(actual, predicted, origin_close):
        actual_direction = np.sign(actual - origin_close)
        predicted_direction = np.sign(predicted - origin_close)

        return (predicted_direction == actual_direction).mean()

    by_horizon_pf = eval.groupby("horizon").apply(
            lambda group: pd.Series({
                "count": len(group),
                "mae": (group['close'] - group['predictions']
                ).abs().mean(),
                "rmse": np.sqrt(
                    (group['close'] - 
                     group['predictions']
                    ).pow(2).mean()
                ),
                "directional_accuracy": (
                    directional_accuracy(group['close'], group['predictions'], group['origin_close'])
                ),
                })
            , include_groups=False) # type: ignore

    mae: float = error.abs().mean()
    rmse: float = error.pow(2).mean() ** 0.5

    # Predicting every close as the origin_close.
    baseline_error: pd.Series = origin_close - actual
    baseline_mae = baseline_error.abs().mean()

    directional_acc = directional_accuracy(actual, predicted, origin_close)

    # Measures how much the model improves upon the baseline.
    mae_skill = 1 - mae / baseline_mae

    by_horizon_loss: dict[str, pd.Series] = {}
    for q in quantiles:
        row_loss = pd.Series(
            pinball_loss(eval['close'], eval[str(q)], q)
        , index=eval.index, dtype=float)

        # row_baseline_loss = pd.Series(
        #         pinball_loss(eval['close'], eval[f"baseline_{q:g}"], q)
        # , index=eval.index, dtype=float)

        by_horizon_loss[str(q)] = row_loss.groupby(eval['horizon']).mean()
        # by_horizon_loss[f"baseline_{q:g}"] = row_baseline_loss.groupby(eval['horizon']).mean()
        # by_horizon_loss[f"{q:g}_skill"]= 1 - (by_horizon_loss[str(q)] / by_horizon_loss[f"baseline_{q:g}"].replace(0, np.nan))

    by_horizon_loss_df = pd.DataFrame(by_horizon_loss)

    # Overall quantile loss.
    loss: dict[float, float] = {}
    baseline_loss: dict[float, float] = {}
    for quantile in quantiles:
        loss[quantile] = pinball_loss(eval['close'], eval[str(quantile)], quantile).mean()
        # baseline_loss[quantile] = pinball_loss(actual, eval[f'baseline_{quantile:g}'], quantile).mean()

    def horizon_interval_metrics(group):
        y_true = group['close']
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

    # Total coverage across all horizons.
    total_coverage: float = (
        (eval[str(prediction_interval.lower)] <= eval['close'])
        & (eval['close'] <= eval[str(prediction_interval.upper)])
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
        ts_eval['close'], 
        by_horizon_pf,
        mae, 
        baseline_mae, 
        mae_skill, 
        rmse, 
        directional_acc,
        by_horizon_loss_df,
        loss, 
        baseline_loss,
        intervals_by_horizon,
        total_coverage,
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

    forecasts = forecasts.sort_values(["ticker", "forecast_origin"]).reset_index(
        drop=True
    )
    outcomes = market_data[['ticker', 'model_timestamp', 'open', 'close']].rename(
            columns={"open": "entry_price", "close": "exit_price"}
    )
    forecasts = forecasts.merge(
            outcomes,
            on=['ticker', 'model_timestamp'],
            how="left",
            validate="one_to_one"
    )

    forecasts["threshold"] = pd.Series(
        calculate_cost_threshold(price, total_round_trip_cost, contract_spec.multiplier)
        for price in forecasts["origin_close"]
    )

    forecasts['signal'] = pd.Series(
            # calcuate_signal_pi(row['threshold'], row[str(pred_interval.lower)], row[str(pred_interval.upper)])
            calculate_signal_point(row['threshold'], row['predictions'] - row['origin_close'])
            for _, row in forecasts.iterrows()
    )

    print(forecasts['signal'].value_counts())
    for idx, row in forecasts.iterrows():
        # print(f"Threshold: {row['threshold']:.8f}. PI: [{row[str(pred_interval.lower)]}, {row[str(pred_interval.upper)]}]. Signal is {row['signal']}")
        print(
            f"Threshold: {row['threshold']:.8f}. Predicted close: {row['predictions']:.8f}. "
            f"Actual close: {row['exit_price']:.8f}. AE: {abs(row['exit_price'] - row['predictions'])}. "
            f"Signal is {row['signal']}"
        )

    forecasts["gross_pnl"] = (
        forecasts["signal"]
        * (forecasts["exit_price"] - forecasts["entry_price"])
        * contract_spec.multiplier
    )

    forecasts['trading_cost'] = np.where(
            forecasts['signal'] != Signal.HOLD,
            total_round_trip_cost,
            0.0
    )

    forecasts['net_pnl'] = forecasts['gross_pnl'] - forecasts['trading_cost']

    trades = forecasts[forecasts['signal'] != Signal.HOLD]
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
    return total_round_trip_cost / multiplier


def calcuate_signal_pi(threshold, lower_b, upper_b) -> Signal:
    if lower_b > threshold:
        signal = Signal.LONG
    elif upper_b < -threshold:
        signal = Signal.SHORT
    else:
        signal = Signal.HOLD
    return signal

def calculate_signal_point(threshold, diff):
    if diff > threshold:
        signal = Signal.LONG
    elif diff < -threshold:
        signal = Signal.SHORT
    else:
        signal = Signal.HOLD
    return signal
