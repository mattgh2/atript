from ..typedefs import EvaluationResult, PredictionInterval
from chronos import Chronos2Pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
from ..figs import plot_predictions

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
        ts_eval[str(prediction_interval.lower)],
        ts_eval[str(prediction_interval.upper)],
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

    plot_predictions(
        eval_results.y_pred,
        eval_results.y_true,
        eval_results.lower_p,
        eval_results.upper_p,
    )

def predict_and_evaluate(
    pipeline,
    context_df,
    history_df,
    pred_length,
    context_length,
    quantiles,
    prediction_interval,
    num_windows: int | None = None,
) -> EvaluationResult:
    e = walk_forward_predict(pipeline, 
            context_df, 
            pred_length, 
            pred_length, 
            context_length, 
            quantiles,
            num_windows
    )
    e = e.merge(
                history_df[['model_timestamp','real_timestamp', 'ticker']],
                on=['ticker', "model_timestamp"],
                how="inner"
        )
    e['context_length'] = context_length

    return evaluate(e, quantiles, prediction_interval)

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
    quantiles: list[float],
    num_windows: int | None = None,
) -> pd.DataFrame:

    if context_df['ticker'].nunique() != 1:
        raise ValueError("Walk-forward prediction requires exactly one ticker.")
    results: list[pd.DataFrame] = []

    last = context_df.shape[0] - horizon
    for window_len in tqdm(range(context_length + 1, last + 1, step)):

        if num_windows is not None and num_windows <= 0:
            break

        label_rows = context_df.iloc[window_len : window_len + horizon]

        '''If any of y_true values in the prediction window are nan, 
           they need to be excluded from evaluation or they will interfere with evaluation metric.'''
        if (
            pd.isna(context_df.iloc[window_len - 1]['close']) or
            label_rows["close"].isna().any()
            or not label_rows["eval_target_ok"].eq(1).all()
        ):
            continue

        model_context = context_df.iloc[:window_len].drop(
                columns=["eval_target_ok"],
                errors="ignore",
        )

        pred_df = predict_chronos(
            pipeline, model_context, horizon, context_length, quantiles
        )

        pred_df["horizon"] = np.arange(1, len(pred_df) + 1)
        pred_df['forecast_origin'] = model_context.iloc[window_len - 1]['model_timestamp'] 
        pred_df['origin_close'] = model_context.iloc[window_len - 1]['close']

        # Compute each quantiles baseline value for the current context window. (historical quantile)
        for quantile in quantiles:
            baseline_value = pred_df['origin_close']
            pred_df[f"baseline_{quantile:g}"] = baseline_value

        eval_df = context_df[['ticker','model_timestamp', 'close']].iloc[window_len: window_len + horizon]
        results.append(pred_df.merge(
                eval_df,
                on=['ticker', 'model_timestamp'],
                how="inner",
                validate="one_to_one"
        ))

        if num_windows is not None:
            num_windows -= 1

    df = pd.concat(results, ignore_index=True)
    return df
