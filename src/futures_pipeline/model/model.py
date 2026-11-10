import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from pathlib import Path
from chronos import Chronos2Pipeline
from ..config import PROCESSED_DATA_DIR, TRAINING_DATA_DIR, MODEL_DIR, FIGURE_PATH, PROJECT_ROOT, FINETUNED_PATH
from ..typedefs import EvaluationResult, TradingFees, ContractSpec, Signal, PredictionInterval, BacktestResults
from ..datareader import load_prior_data, fetch_contract_spec
from ..datareader.readerutil import get_product_code
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ..config import load_fees, WEB_PATH
from datetime import timedelta
from transformers import EarlyStoppingCallback
from typing import NamedTuple, cast
from tqdm import tqdm
from ..figs import plot_predictions, parameter_performance
from ..web_util import web_scaffolding, display_figs_on_page, update_resolution_page, write_training_results
from logging import Logger
from collections.abc import Sequence, Mapping
from .backtest import backtest_strategy
from .evaluate import evaluate, report_eval, predict_and_evaluate, walk_forward_predict
from .features import *

Log = Logger(name='log', level=1)

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
    context_length: int, pred_length: int
) -> TrainTestValidate:

    if not 0 < train_size < 1:
        raise ValueError("train_size must be in the range (0, 1).")

    product_code = get_product_code(symbol)

    train_df: pd.DataFrame = load_prior_data(
        symbol=product_code, 
        resolution=resolution,
        data_dir = PROCESSED_DATA_DIR
    )

    if train_df.empty:
        raise RuntimeError("Missing training set.")

    train_df = train_df.sort_values(['ticker','real_timestamp'])

    train_df['session_end_date'] = pd.to_datetime(train_df['session_end_date'])

    # Hold out the final contract for evaluation. 
    # The routine to fetch training data ensures that the final contract is complete (ends on a roll-over).
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
        data_dir=PROCESSED_DATA_DIR, 
    )
    if history_df.empty:
        raise RuntimeError("Missing context set.")

    return history_df


def build_training_input(train_df, covariates: list[str], pred_length) -> list:
    train = []

    # Contruct one training item per ticker.
    for ticker, contract_df in train_df.groupby('ticker', sort=False):

        # Ensure sorted by ascending timestamp and drop examples with missing labels.
        contract_df = contract_df.sort_values("real_timestamp").reset_index(drop=True)

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
) -> tuple[Sequence[Mapping], Sequence[Mapping]]:

    """ Note: A validation item results in a single forecast window. 
              Need to futher split each item into multiple validation windows."""              

    """ Note: Forecast windows containing nan should be dropped. """

    val = []
    contract_items: dict[str, list[tuple[dict, dict]]] = {}
    for ticker, contract_df in val_df.groupby('ticker', sort=False):
        ticker = cast(str, ticker)
        # Ensure sorted by ascending timestamp.
        contract_df = contract_df.sort_values("real_timestamp").reset_index(drop=True)

        # Min amount of examples for validation.
        if contract_df.shape[0] < context_length + pred_length:
            Log.warning(f"ticker is not large enough for validation.")
            continue

        # Overlap historical context with nonoverlapping prediction windows.
        # Start at the end to prioritize the most recent data.
        forecast_origins = np.arange(
            len(contract_df) - pred_length, context_length - 1, -pred_length
        )[::-1]

        for origin in forecast_origins:
            window = contract_df.iloc[origin - context_length: origin + pred_length]

            # Dont include the window if there are nan's in the prediction rows.
            # This is interfere with evaluation metrics.
            if (
                window["close"].tail(pred_length + 1).isna().any()
                or not window.tail(pred_length)["eval_target_ok"].eq(1).all()
            ):
                continue

            past_covariates = {
                column: np.array(values)
                for column, values in window[[*covariates]].to_dict(orient="list").items()
            }
            contract_items.setdefault(ticker, []).append(
                (
                    {
                        "target": window["close"].to_numpy(),
                        "past_covariates": past_covariates,
                        # "future_covariates": {} # TODO: load future time covariates
                    },
                    {
                        "context": window.iloc[:-pred_length],
                        "actual": window[['ticker', 'close', 'model_timestamp', 'real_timestamp']].tail(pred_length),
                        "origin": origin,
                    }
                )
            )

    if len(contract_items) == 0:
        raise ValueError(f"No contract has a sufficent length for validation.")

    min_origins = min(len(item) for item in contract_items.values())

    # Reduce length of each item set down to min_origins.
    for contract, items in contract_items.items():
        items[:] = items[-min_origins:]

    # Flatten the (val_item, window) pairs.
    items = [
            item for items in contract_items.values() 
            for item in items
    ]

    val_items, windows = zip(*items)

    if not val_items or not windows:
        raise ValueError("No contracts contain enough validation observations.")

    return (val_items, windows)


def run_model(
    ticker,
    resolution,
    pred_length,
    context_length,
    quantiles,
    prediction_interval: PredictionInterval,
    store_weights: bool,
    hf_token=None,
    train: bool = False,
    zero_shot: bool = False,
    eval: bool=False,
    tune: bool = False,
    _abalation: bool = False
) -> None:

    if _abalation:
        dataset = load_and_split_training_set(
            train_size=0.90,
            symbol=ticker, 
            resolution=resolution, 
            context_length=context_length, 
            pred_length=pred_length
        )

        from ..abalation import abalation
        abalation(dataset, pred_length, context_length, quantiles, prediction_interval, resolution)
        return

    covariates: list = [*VOLATILITY, *TREND, *MOMENTUM, *VOLUME, *TIME, *MISSING]

    # Get the trained model directory.
    output_dir = MODEL_DIR / f"{get_product_code(ticker)}-{resolution}"
    checkpoint_dir = output_dir / "finetuned-ckpt"

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
        validation_input, _ = build_validation_input(validate_df, covariates, pred_length, context_length)

        pipeline = finetune_chronos(
            pipeline=pipeline,
            train=training_input,
            validation=validation_input,
            pred_length=pred_length,
            context_length=context_length,
            finetune_mode="lora",
            learning_rate=1e-5,
            num_steps=1000,
            batch_size=20,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            eval_steps=100,
            save_steps=100,
            model_dir=output_dir,
            store_weights=store_weights
        )

        timestamps = test_df[['model_timestamp', 'real_timestamp', 'ticker']]

        market_data = test_df[['ticker', 'open', 'close', 'model_timestamp']].copy()
        test_df = test_df[['model_timestamp', 'ticker', 'eval_target_ok', 'close', *covariates]]

        test_set_ticker = test_df['ticker'].iloc[0]

        # Note: walk_forward_predict excludes eval_target_ok from its prediction context.
        forecasts = walk_forward_predict(
            pipeline, 
            test_df, 
            pred_length, 
            pred_length, 
            context_length, 
            quantiles
        )

        e = forecasts.merge(
                timestamps,
                on=['ticker', "model_timestamp"],
                how="inner"
        )

        eval_results = evaluate(e, quantiles, prediction_interval)
        contract_spec: ContractSpec = fetch_contract_spec(test_set_ticker)


        _, backtest_results = backtest_strategy(
            forecasts, 
            market_data, 
            prediction_interval, 
            contract_spec, 
            pred_length
        )

        results_dir: Path = FINETUNED_PATH / resolution
        results_dir.mkdir(exist_ok=True, parents=True)

        update_resolution_page(
                FINETUNED_PATH / 'trained.html',
                resolution,
                results_dir / 'results.html',
                FINETUNED_PATH / 'nav.html' 
        )

        write_training_results(
            results_dir / 'results.html',
            eval_results,
            backtest_results,
            []
        )
        return


    history_df = load_history(ticker, resolution)

    context_df = (
        history_df[["model_timestamp", "ticker", "close", *covariates]]
        .sort_values(['ticker', "model_timestamp"])
        .reset_index(drop=True)
    )

    if eval:
        context_df['eval_target_ok'] = history_df['eval_target_ok']
        if tune:
            # context_lengths = [2 ** n for n in range(5, 14)]
            context_lengths = [2 ** n for n in range(5, 7)]
            tune_context_length(
                    pipeline,
                    context_lengths,
                    context_df,
                    history_df,
                    pred_length,
                    quantiles,
                    prediction_interval,
                    zero_shot,
                    resolution
            )
        else:
            eval_results = predict_and_evaluate(
                    pipeline,
                    context_df,
                    history_df,
                    pred_length,
                    context_length,
                    quantiles,
                    prediction_interval,
            )

            report_eval(eval_results)


"""
Loads a Chronos model from local storage or Hugging Face.

@param model_dir the name of the directory where the model is located.
@param store_weights Boolean flag for storing weights locally.
@param hf_token A Hugging Face token.

@returns A ``Chronos2Pipeline``
"""
def load_chronos(
    model_dir: Path | None = None,
    hf_token: str | None = None,
) -> Chronos2Pipeline:
    if model_dir and not model_dir.exists():
        raise ValueError(f"{model_dir} does not exist.")
    if model_dir:
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



# Add button for resolution if it does not exist. May require some tree traversal.
# Can dominate be used to parse html? -- BeautifulSoup
# parse the list of resolution divs in enclosing div with id="resolutions".
# html file -> zeroshot_context_lengths.html

# TODO: Remove hardcoding paths.
def tune_context_length(
    pipeline: Chronos2Pipeline,
    param_values: list[int],
    context_df: pd.DataFrame,
    history_df: pd.DataFrame,
    pred_length: int,
    quantiles: list[float],
    prediction_interval: PredictionInterval,
    zero_shot: bool,
    resolution: str
) -> None:


    base_path = WEB_PATH / ("zeroshot" if zero_shot else "finetuned") / f"hyperparameter_tuning"
    update_resolution_page(
        base_path / "context_lengths.html",
        resolution,
        base_path / resolution / "predictions.html",
        Path("hyperparameters.html"),
    )
    update_resolution_page(
        base_path / "context_length_metrics.html",
        resolution,
        base_path / resolution / "metrics.html",
        Path("hyperparameters.html"),
    )

    # Ensure the same number of prediction windows are used for each parameter value.
    num_pred_windows = (context_df.shape[0] - max(param_values)) // pred_length

    mae_scores = []
    directional_acc = []
    figs: list[tuple[Path, list[str]]] = []

    fig_path: Path = FIGURE_PATH / f"hyperparam_optimization/inference/"
    fig_path /= ("zeroshot" if zero_shot else "finetuned") 
    fig_path /= resolution

    fig_path.mkdir(exist_ok=True, parents=True)

    save_path: Path = (
        WEB_PATH / (f"zeroshot/" if zero_shot else f"finetuned/") / f"hyperparameter_tuning/{resolution}"
    )

    save_path.mkdir(parents=True, exist_ok=True)

    for context_length in param_values:
        eval_results: EvaluationResult = predict_and_evaluate(
            pipeline,
            context_df,
            history_df,
            pred_length,
            context_length,
            quantiles,
            prediction_interval,
            num_pred_windows,
        )

        current_output: list[str] = []

        mae_scores.append(eval_results.mae)
        directional_acc.append(eval_results.directional_accuracy)

        current_output.append(f"context length: {context_length}")
        current_output.append(
            f"directional accuracy: {eval_results.directional_accuracy:.2f}"
        )
        current_output.append(f"MAE: {eval_results.mae:.2f}")
        current_output.append(
            f"MAE improvement over baseline: {eval_results.mae_skill:.2%}"
        )
        current_output.append(f"PI coverage: {eval_results.pi_coverage:.2%}")
        current_output.append(f"Mean interval width: {eval_results.mean_interval_width}")

        current_output.append("Quantile calibration:")
        for q, v in eval_results.quantile_calibration.items():
            current_output.append(f"\tq={q}: {v}")

        current_output.append(f"Calibration Error:")
        for q, v in eval_results.calibration_error.items():
            current_output.append(f"\tq={q}: {v}")

        current_fig_path = fig_path / f"{context_length}-predictions.png"

        plot_predictions(
            eval_results.y_pred.head(200),
            eval_results.y_true.head(200),
            eval_results.lower_p.head(200),
            eval_results.upper_p.head(200),
            current_fig_path,
        )

        figs.append((current_fig_path, current_output))

    mae_perf_path = fig_path / "mae-performance.png"
    parameter_performance(
            "Context Length",
            "MAE",
            param_values,
            mae_scores,
            mae_perf_path
    )

    dir_acc_perf_path = fig_path / "directional-performance.png"
    parameter_performance(
            "Context Length",
            "Directional Accuracy",
            param_values,
            directional_acc,
            dir_acc_perf_path
        )

    save_predictions_file = save_path / "predictions.html"
    save_metrics_file: Path = save_path / "metrics.html"
    display_figs_on_page(
        figs,
        save_predictions_file,
        base_path / "context_lengths.html",
    )

    display_figs_on_page(
        [(mae_perf_path, []), (dir_acc_perf_path, [])],
        save_metrics_file,
        base_path / "context_length_metrics.html",
    )
