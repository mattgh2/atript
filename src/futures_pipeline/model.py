import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from pathlib import Path
from chronos import Chronos2Pipeline
from .config import PROCESSED_DATA_DIR, TRAINING_DATA_DIR, MODEL_DIR
from .typedefs import EvaluationResult, TradingFees, ContractSpec, Signal, PredictionInterval, BacktestResults
from .datareader import load_prior_data, fetch_contract_spec
from .datareader.readerutil import get_product_code
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import load_fees
from datetime import timedelta
from transformers import EarlyStoppingCallback
from typing import NamedTuple
from tqdm import tqdm
from .figs import plot_predictions

TREND = ['EMA', 'SMA', 'DMA', 'T3MA']
MOMENTUM = ["RSI", 'FSO', 'SSO', 'ROC', 'CCI']
VOLATILITY = ["VR", "ATR", "percent_b"]
VOLUME = ["volume", "VWAP"]
TIME = ['has_time_gap', 'log_elapsed_intervals']

# VOLATILITY = [
#     "VR",
#     "ATR",
#     "percent_b",
#     "upper_b",
#     "lower_b",
#     "upper_kc",
#     "lower_kc",
#     "bollinger_squeeze",
# ]

# TODO: (1) Should each contract have the same number of validation items?
#           More items contribute more to loss thus resulting in unequal ticker contributions.
#       (2) In build_validation_set, reduce context_length if too large. bsearch?
#       (3) Tune Context length for training (4) Find optimal context length for inference.
#       (5) For evaluation, use the remainder of len(context) - context_length. If context_length is very small
#           e.g. context_lenth = 256, then this may not be reasonable.

class TrainTestValidate(NamedTuple):
    train: pd.DataFrame
    test: pd.DataFrame
    validate: pd.DataFrame

def load_and_split_training_set(
    train_size: float, 
    symbol: str, 
    resolution: str, 
    context_length: int,
    pred_length: int
) -> TrainTestValidate:

    if not 0 < train_size < 1:
        raise ValueError("train_size must be in the range (0, 1).")

    product_code = get_product_code(symbol)

    train_df: pd.DataFrame = load_prior_data(
        symbol=product_code, 
        resolution=resolution,
        data_dir = PROCESSED_DATA_DIR / f"{symbol}-{resolution}"
    )

    if train_df.empty:
        raise RuntimeError("Missing training set.")

    train_df = train_df.sort_values(['ticker','real_timestamp'])

    train_df['session_end_date'] = pd.to_datetime(train_df['session_end_date'])

    # Hold out the final contract for evaluation.
    final_contract = train_df.loc[
        train_df["session_end_date"].eq(train_df["session_end_date"].max()), "ticker"
    ].iloc[0]

    test_df = train_df[train_df['ticker'].eq(final_contract)]
    train_df = train_df[train_df['ticker'].ne(final_contract)]


    val_dfs: list[pd.DataFrame] = []
    train_dfs: list[pd.DataFrame] = []

    for ticker, contract_df in train_df.groupby('ticker', sort=False):
        split_idx: int = int(contract_df.shape[0] * train_size)

        # This check ensures that there is atleast one valid training window.
        if split_idx < max(context_length, 2 * pred_length):
            continue

        # Need at least pred_length observations for validation.
        # This ensures at least one valid validation window.
        if contract_df.shape[0] - split_idx < pred_length:
            continue

        train_dfs.append(contract_df.iloc[:split_idx])
        val_dfs.append(contract_df.iloc[split_idx - context_length:])

    if not train_dfs or not val_dfs:
        raise ValueError("not enough training data for the provided context_length.")

    validation_df = pd.concat(val_dfs, ignore_index=True)
    train_df = pd.concat(train_dfs, ignore_index=True)

    return TrainTestValidate(train_df, test_df, validation_df)


def load_history(ticker: str, resolution: str) -> pd.DataFrame:
    history_df: pd.DataFrame = load_prior_data(
        ticker, 
        resolution,
        data_dir=PROCESSED_DATA_DIR / f"{ticker}-{resolution}", 
    )
    if history_df.empty:
        raise RuntimeError("Missing context set.")

    return history_df


def build_training_input(train_df, covariates: list[str], pred_length) -> list:
    train = []

    # Contruct one training item per ticker.
    for ticker, contract_df in train_df.groupby('ticker', sort=False):

        # Ensure sorted by ascending timestamp and drop examples with missing labels.
        contract_df = contract_df.sort_values("real_timestamp").reset_index(drop=True).dropna(
            subset=['close']
        )

        # Needs at least pred_length historical examples.
        if contract_df.shape[0] < 2 * pred_length:
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

def build_validation_input(
    val_df: pd.DataFrame, 
    covariates: list[str], 
    pred_length: int, 
    context_length: int
) -> list:

    """ Note: A validation item results in a single forecast window. 
              Need to futher split each item into multiple validation windows."""              

    min_origins = float("inf")
    val = []
    contract_items = {}
    for ticker, contract_df in val_df.groupby('ticker', sort=False):

        # Ensure sorted by ascending timestamp and drop examples with missing labels.
        contract_df = contract_df.sort_values("real_timestamp").reset_index(drop=True).dropna(
            subset=['close']
        )

        # Min amount of examples for validation.
        if contract_df.shape[0] < context_length + pred_length:
            continue

        # Overlap historical context with nonoverlapping prediction windows. 
        # Start at the end to prioritize the most recent data.
        forecast_origins = np.arange(len(contract_df) - pred_length, context_length - 1, -pred_length)[::-1]
        min_origins = min(min_origins, len(forecast_origins))

        for origin in forecast_origins:
            window = contract_df.iloc[origin - context_length: origin + pred_length]
            past_covariates = {
                column: np.array(values)
                for column, values in window[[*covariates]].to_dict(orient="list").items()
            }
            contract_items.setdefault(ticker, []).append(
                {
                    "target": window['close'].to_numpy(),
                    "past_covariates": past_covariates,
                    # "future_covariates": {} # TODO: load future time covariates
                }
            ) 

    # Reduce length of each item set down to min_origins.
    for contract, items in contract_items.items():
        items[:] = items[-min_origins:]

    val = [
            item for items in contract_items.values() 
            for item in items
    ]

    if not val:
        raise ValueError("No contracts contain enough validation observations.")

    return val


def run_model(
    ticker,
    resolution,
    pred_length,
    context_length,
    quantiles,
    prediction_interval: PredictionInterval,
    contract_spec: ContractSpec,
    store_weights: bool,
    hf_token=None,
    train: bool = False,
    zero_shot: bool = False,
    eval: bool=False
) -> None:

    covariates: list = [*VOLATILITY, *TREND, *MOMENTUM, *VOLUME, *TIME]

    model_dir: Path | None = None

    # Get the trained model directory.
    output_dir = MODEL_DIR / f"{get_product_code(ticker)}-{resolution}"
    checkpoint_dir = output_dir / "finetuned_ckpt"

    load_dir = None if train or zero_shot else checkpoint_dir

    pipeline = load_chronos(
        load_dir,
        hf_token=hf_token,
    )

    if train:
        train_df, test_df, validate_df = load_and_split_training_set(
            train_size=0.90,
            symbol=ticker, 
            resolution=resolution, 
            context_length=context_length, 
            pred_length=pred_length
        )

        # Build model inputs.
        training_input = build_training_input(train_df, covariates, pred_length)
        validation_input = build_validation_input(validate_df, covariates, pred_length, context_length)

        pipeline = finetune_chronos(
            pipeline=pipeline,
            train=training_input,
            validation=validation_input,
            pred_length=pred_length,
            context_length=context_length,
            finetune_mode="full",
            learning_rate=1e-6,
            num_steps=1000,
            batch_size=256,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            eval_steps=100,
            save_steps=100,
            model_dir=output_dir,
            store_weights=store_weights
        )

        timestamps = test_df[['model_timestamp', 'real_timestamp']]
        test_df = test_df[['model_timestamp', 'ticker', 'close', *covariates]]

        e = walk_forward_predict(
            pipeline, 
            test_df, 
            pred_length, 
            pred_length, 
            context_length, 
            int(test_df.shape[0] * .80),
            quantiles
        )

        e = e.merge(
                timestamps,
                on="model_timestamp",
                how="inner"
        )

        eval_results = evaluate(e, quantiles, prediction_interval)
        report_eval(eval_results)
        return

    history_df = load_history(ticker, resolution)

    context_df = (
        history_df[["model_timestamp", "ticker", "close", *covariates]]
        .sort_values("model_timestamp")
        .reset_index(drop=True)
    )

    if eval:
        initial_train_size = int(context_df.shape[0] * .80)
        e = walk_forward_predict(pipeline, 
            context_df, 
            pred_length, 
            pred_length, 
            context_length, 
            initial_train_size, 
            quantiles
        )
        e = e.merge(
                history_df[['model_timestamp','real_timestamp']],
                on="model_timestamp",
                how="inner"
        )
        eval_results: EvaluationResult = evaluate(e, quantiles, prediction_interval)
        report_eval(eval_results)


    else:
        print(f"Performing walk-forward forecasting")
        forecasts = walk_forward_predict(
            pipeline,
            context_df,
            1,
            1,
            context_length,
            int(context_df.shape[0] * 0.80),
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
    context_length: int,
    quantiles,
) -> pd.DataFrame:

    # Generate predictions with covariates
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=pred_length,  # Number of steps to forecast
        context_length=context_length,
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
    model_dir: Path | None,
    hf_token: str | None = None,
) -> Chronos2Pipeline:
    if model_dir and model_dir.exists():
        try: 
            return Chronos2Pipeline.from_pretrained(
                model_dir, device_map="cuda", local_files_only=True
            )
        except OSError as e:
            raise ValueError(f"Failed to load model from {model_dir}.") from e

    pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map="cuda", token=hf_token
    )

    return pipeline


def finetune_chronos(
    pipeline: Chronos2Pipeline,
    train,
    validation,
    pred_length: int ,
    context_length: int,
    finetune_mode: str,
    learning_rate: float,
    num_steps: int,
    batch_size: int,
    callbacks: list,
    eval_steps: int,
    save_steps: int,
    model_dir: Path | None = None,
    store_weights: bool = False,
):
    fine_tuning_params = {
            "inputs": train,
            "validation_inputs": validation,
            "prediction_length": pred_length,
            "context_length": context_length,
            "learning_rate": learning_rate,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "finetune_mode": finetune_mode,
            "callbacks": callbacks,
            "eval_steps": eval_steps,
            "save_steps": save_steps
    }

    if store_weights:
        assert model_dir is not None, "model_dir is required when store_weights=True"
        fine_tuning_params['output_dir'] = model_dir

    pipeline = pipeline.fit(**fine_tuning_params)

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
    context_length: int,
    initial_train_size: int,
    quantiles: list[float],
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    last = context_df.shape[0] - horizon
    for window_len in tqdm(range(initial_train_size, last + 1, step)):
        pred_df = predict_chronos(pipeline, context_df[: window_len], horizon, context_length, quantiles)
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

    ts_eval = eval.set_index("real_timestamp")
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

def report_eval(eval_results: EvaluationResult) -> None:
    print(f"=== POINT FORECAST METRICS ===")
    print(eval_results.by_horizon.to_string())
    print(f"MAE: {eval_results.mae:.8f}")
    print(f"Baseline MAE: {eval_results.baseline_mae:.8f}")
    print(f"MAE Skill: {eval_results.mae_skill:.2%}")
    print(f"RMSE: {eval_results.rmse:.8f}")
    print(f"Directional accuracy: {eval_results.directional_accuracy:.8f}\n")

    # loss_skill = {
    #         q: 1 - eval_results.loss[q] / eval_results.baseline_loss[q]
    #         if eval_results.baseline_loss[q] != 0 else np.nan
    #         for q in quantiles
    # }

    print(f"=== QUANTILE LOSS ===")
    print(eval_results.by_horizon_loss.to_string())
    # for t in zip(
    #     eval_results.loss.items(),
    #     eval_results.baseline_loss.items(),
    #     loss_skill.items(),
    # ):
    #     (quantile, avg_loss), (_, baseline_loss), (_, loss_skill) = t
    #     print(f"avg loss for {quantile:.2%}th quantile: {avg_loss:.10f}")
    #     print(f"avg baseline loss for {quantile:.2%}th quantile: {baseline_loss:.10f}")
    #     print(f"loss skill for {quantile:.2%}th quantile: {loss_skill:.2%}\n")

    for quantile, avg_loss in eval_results.loss.items():
        print(f"avg loss for {quantile:.2%}th quantile: {avg_loss:.10f}")

    print(f" === Quantile Calibration === ")
    for q, v in eval_results.quantile_calibration.items():
        print(f"q={q}: {v}")

    print(f" \n=== Calibration Error === ")
    for q, v in eval_results.calibration_error.items():
        print(f"q={q}: {v}")

    print(f"\n=== PREDICTION INTERVAL ===")
    print(eval_results.intervals_by_horizon.to_string())
    print(f"PI coverage: {eval_results.pi_coverage:.2%}")
    print(f"Nominal coverage: {eval_results.nominal_coverage:.2%}")
    print(f"Mean PI width: {eval_results.mean_interval_width:.8f}")

    plot_predictions(np.arange(len(eval_results.y_pred)), eval_results.y_pred, eval_results.y_true)

