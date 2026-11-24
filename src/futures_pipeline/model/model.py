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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ParameterSampler
from scipy.stats import loguniform
from itertools import chain
from .evaluate import NoValidWindowsError
from typing import Literal

# Change the evaluation metrics to use the quantiles instead of the point forecast to assess model strength.

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
    tune_mode: Literal['None', 'zero_shot', 'finetuned'] = "None",
    _abalation: bool = False
) -> None:
    if tune_mode != 'None' and train:
        raise ValueError("Cannot pass --train with --tune.")

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

    covariates: list = [
        *TRADING_ACTIVITY,
        *PRICE_LOCATION,
        *SESSION_TIMING,
        *TIME_GAPS,
        *MISSING_FLAGS,
        *NEW_VOLATILITY,
        *RECENT_RETURNS,
        *NEW_TREND,
        *CANDLE_STRUCTURE,
    ]

    # covariates: list = [*VOLATILITY, *TREND, *MOMENTUM, *VOLUME, *TIME, *MISSING]

    if tune_mode == 'finetuned':
        num_features = 1 + len(covariates)
        param_space = {
                'context_length': [128, 256, 512, 1024, 2048],
                'learning_rate': loguniform(1e-7, 3e-5),
                'batch_size': [num_features * x for x in [4,8,16]],
        }

        product_code = get_product_code(ticker)

        train_df = load_prior_data(
            symbol=product_code, 
            resolution=resolution,
            data_dir = PROCESSED_DATA_DIR
        )
    
        if train_df.empty:
            raise RuntimeError("Missing training set.")
    
        train_df = train_df.sort_values(['ticker','real_timestamp'])
    
        train_df['session_end_date'] = pd.to_datetime(train_df['session_end_date'])

        hyperparam_tuning(
                train_df,
                param_space,
                pred_length,
                resolution,
                covariates,
                quantiles,
                prediction_interval,
                n_iter=10,
                hf_token=hf_token
        )
        return


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
            batch_size=len(covariates) + 1,
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
        history_df[["model_timestamp", "ticker", "close", 'eval_target_ok', *covariates]]
        .sort_values(['ticker', "model_timestamp"])
        .reset_index(drop=True)
    )
    timestamps = history_df[['model_timestamp', 'real_timestamp', 'ticker']]

    if tune_mode == 'zero_shot':
        context_lengths = [2 ** n for n in range(5, 14)]
        # context_lengths = [2 ** n for n in range(5, 7)]
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
        forecasts = walk_forward_predict(
                pipeline,
                context_df,
                pred_length,
                pred_length,
                context_length,
                quantiles,
        )

        e = forecasts.merge(
                timestamps,
                on=['ticker', "model_timestamp"],
                how="inner"
        )

        eval_results = evaluate(e, quantiles, prediction_interval)
        contract_spec = fetch_contract_spec(ticker)

        _, backtest_results = backtest_strategy(
            forecasts, 
            history_df, 
            prediction_interval, 
            contract_spec, 
            pred_length
        )

        report_eval(eval_results)
        backtest_results.display_results()



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

    if len(context_df['ticker'].unique()) != 1:
        raise ValueError("Multi-ticker context is not supported.")
    ticker = context_df['ticker'].iloc[0]

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
    net_sharpes = []
    figs: list[tuple[Path, Sequence[str]]] = []

    fig_path: Path = FIGURE_PATH / f"hyperparam_optimization/inference/"
    fig_path /= ("zeroshot" if zero_shot else "finetuned") 
    fig_path /= resolution

    fig_path.mkdir(exist_ok=True, parents=True)

    save_path: Path = (
        WEB_PATH / (f"zeroshot/" if zero_shot else f"finetuned/") / f"hyperparameter_tuning/{resolution}"
    )

    save_path.mkdir(parents=True, exist_ok=True)

    for context_length in param_values:

        forecasts = walk_forward_predict(
            pipeline, 
            context_df, 
            pred_length, 
            pred_length, 
            context_length, 
            quantiles,
            num_windows=num_pred_windows,
            start_pred=max(param_values)
        )

        e = forecasts.merge(
                history_df[['real_timestamp', 'model_timestamp', 'ticker']],
                on=['ticker', "model_timestamp"],
                how="inner"
        )

        eval_results = evaluate(e, quantiles, prediction_interval)
        contract_spec: ContractSpec = fetch_contract_spec(ticker)

        _, backtest_results = backtest_strategy(
            forecasts, 
            history_df, 
            prediction_interval, 
            contract_spec, 
            pred_length
        )

        current_output: list[str] = []

        net_sharpes.append(backtest_results.net_sharpe)
        mae_scores.append(eval_results.mae)
        directional_acc.append(eval_results.directional_accuracy)

        current_output.append(f"context length: {context_length}")
        current_output.extend([*backtest_results.to_strings(), '\n', *eval_results.to_strings()])

        current_fig_path = fig_path / f"{context_length}-predictions.png"

        plot_predictions(
            eval_results.y_pred.head(200),
            eval_results.y_true.head(200),
            eval_results.lower_p.head(200),
            eval_results.upper_p.head(200),
            current_fig_path,
        )

        figs.append((current_fig_path, current_output))

    net_sharpe_perf_path = fig_path / "net_sharpe-performance.png"
    parameter_performance(
            "Context Length",
            "net sharpe",
            param_values,
            net_sharpes,
            net_sharpe_perf_path
    )

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
        [(mae_perf_path, []), (dir_acc_perf_path, []), (net_sharpe_perf_path, [])],
        save_metrics_file,
        base_path / "context_length_metrics.html",
    )


# Randomized search. Walk forward CV for each configuration.
# Hold out a contract for final configuration evaluation.
def hyperparam_tuning(
    train_df,
    param_space: dict[str, list],
    pred_length,
    resolution,
    covariates,
    quantiles,
    prediction_interval,
    n_iter,
    hf_token=None,
):
    base_path = (
        FINETUNED_PATH
        / "hyperparameter_tuning"
    )

    base_pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2", device_map="cuda", token=hf_token
    )

    # Hold out final contract for evaluation.
    final_contract = train_df.loc[
        train_df["session_end_date"].eq(train_df["session_end_date"].max()), "ticker"
    ].iloc[0]

    test_df = train_df.loc[
            train_df['ticker'].eq(final_contract)
    ]
    train_df = train_df.loc[
            train_df['ticker'].ne(final_contract)
    ]

    row_month = pd.to_datetime(
            train_df['session_end_date']
    ).dt.to_period("M")

    months: np.ndarray = pd.Index(row_month.unique()).sort_values().to_numpy()

    splitter = TimeSeriesSplit(n_splits=3, test_size=1)

    candidates = ParameterSampler(
            param_space,
            n_iter=n_iter,
            random_state=42
    )

    pred_origin_start = max(param_space['context_length'])

    figs: list[tuple[Path, Sequence[str]]] = []
    for count, params in enumerate(candidates):
        prediction_dfs: list[pd.DataFrame] = []
        backtest_dfs: list[pd.DataFrame] = []
        total_missing = 0
        for train_idx, val_idx in splitter.split(months):

            fold_train_df = train_df.loc[row_month.isin(months[train_idx[:-1]])]
            finetune_val_df = train_df.loc[row_month.eq(months[train_idx[-1]])]
            val_df = train_df.loc[row_month.isin(months[val_idx])]

            training_inputs = build_training_input(
                    fold_train_df, covariates, pred_length
            )

            validation_inputs, _ = build_validation_input(
                finetune_val_df,
                covariates,
                pred_length,
                params['context_length']
            )

            fitted_pipeline = base_pipeline.fit(
                    inputs=training_inputs,
                    validation_inputs=validation_inputs,
                    prediction_length=pred_length,
                    context_length=params['context_length'],
                    learning_rate=params['learning_rate'],
                    num_steps=1000,
                    batch_size=params['batch_size'],
                    finetune_mode='full',
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            )

            # val_df month could be on a roll-month, in which case there
            # is more than one ticker present in the DataFrame. Need to group by ticker.
            prediction_df: pd.DataFrame = pd.DataFrame()
            backtest_df: pd.DataFrame = pd.DataFrame()
            for ticker, contract_df in val_df.groupby('ticker', sort=False):
                contract_spec = fetch_contract_spec(ticker)

                extra_context_df: pd.DataFrame = train_df.loc[row_month.isin(months[train_idx])]
                ticker_rows = extra_context_df.loc[extra_context_df['ticker'].eq(ticker)]
                extra_context_df = ticker_rows.sort_values("real_timestamp").tail(
                    pred_origin_start
                )

                contract_df = (
                    pd.concat([contract_df, extra_context_df], ignore_index=True)
                    .sort_values("real_timestamp")
                    .reset_index(drop=True)
                )

                context_df = (
                    contract_df[
                        [
                            "ticker",
                            "model_timestamp",
                            "close",
                            "eval_target_ok",
                            *covariates,
                        ]
                    ]
                    .sort_values("model_timestamp")
                    .reset_index(drop=True)
                )

                try:
                     preds = walk_forward_predict(
                         fitted_pipeline,
                         context_df,
                         pred_length,
                         pred_length,
                         params["context_length"],
                         quantiles,
                         start_pred=pred_origin_start
                     )
                except NoValidWindowsError:
                    Log.warning('Skipping %s: no valid scoring windows', ticker)
                    continue

                bt_df, summary  = backtest_strategy(
                        preds,
                        contract_df,
                        prediction_interval,
                        contract_spec,
                        pred_length
                )
                total_missing += summary.missing_count

                backtest_df = pd.concat([backtest_df, bt_df], ignore_index=True)
                prediction_df = pd.concat([prediction_df, preds], ignore_index=True)

            if prediction_df.empty:
                raise ValueError(f'No contracts have valid scoring windows for {months[val_idx[0]]}')

            prediction_df = prediction_df.sort_values('model_timestamp').reset_index(drop=True)
            backtest_df = backtest_df.sort_values('model_timestamp').reset_index(drop=True)

            prediction_df = prediction_df.merge(
                    train_df[['model_timestamp', 'real_timestamp', 'ticker']],
                    on=['ticker', 'model_timestamp'],
                    how='inner',
                    validate='one_to_one'
            )

            prediction_dfs.append(prediction_df)
            backtest_dfs.append(backtest_df)

        all_predictions = pd.concat(
                prediction_dfs,
                ignore_index=True
        )
        signal_ledger = pd.concat(
                backtest_dfs,
                ignore_index=True
        )

        eval_results = evaluate(
                all_predictions,
                quantiles,
                prediction_interval
        )

        trade_ledger = signal_ledger.loc[
                signal_ledger['signal'].ne(0)
        ]

        allocated_capital: float = 1000.0
        risk_free_per_window: float = 0.0
        returns: pd.Series = signal_ledger['net_pnl'] / allocated_capital
        excess: pd.Series = returns - risk_free_per_window
        std = excess.std(ddof=1)
        net_sharpe_per_window = excess.mean() / std if len(excess) >= 2 and std > 0 else float('nan')

        backtest_summary = BacktestResults(
            trade_count=len(trade_ledger),
            net_pnl=trade_ledger["net_pnl"].sum(),
            win_rate=trade_ledger["net_pnl"].gt(0).mean(),
            average_pnl=trade_ledger["net_pnl"].mean(),
            gross_profit=trade_ledger.loc[
                trade_ledger["net_pnl"].gt(0), "net_pnl"
            ].sum(),
            gross_loss=-trade_ledger.loc[
                trade_ledger["net_pnl"].lt(0), "net_pnl"
            ].sum(),
            missing_count=total_missing,
            net_sharpe=net_sharpe_per_window
        )

        output: list[str] = list(
            chain(
                [
                    "Configuration:",
                    *[
                        f"{param_name}: {param_value}"
                        for param_name, param_value in params.items()
                    ],
                ],
                eval_results.to_strings(),
                ["\n"],
                backtest_summary.to_strings(),
            )
        )
        first_contract = all_predictions.loc[
                all_predictions['real_timestamp'].idxmin(), 'ticker'
        ]

        plot_df = all_predictions.loc[all_predictions['ticker'].eq(first_contract)].head(200)
        plot_y_pred = plot_df['predictions'].head(200)
        plot_y_true = plot_df['close'].head(200)
        plot_lower = plot_df[str(prediction_interval.lower)].head(200)
        plot_upper = plot_df[str(prediction_interval.upper)].head(200)

        
        fig_path: Path = FIGURE_PATH / f"abalation/finetuned/{resolution}"
        fig_path.mkdir(exist_ok=True, parents=True)
        current_fig_path = fig_path / f"{count}-predictions.png"


        plot_predictions(
            plot_y_pred,
            plot_y_true,
            plot_lower,
            plot_upper,
            current_fig_path,
        )

        figs.append((current_fig_path, output))

    update_resolution_page(
        base_path / "randomized_search.html",
        resolution,
        base_path / f"{resolution}/results.html",
        base_path / "hyperparameters.html"
    )
    (base_path / resolution).mkdir(exist_ok=True, parents=True)

    display_figs_on_page(
        figs, 
        base_path / resolution / "results.html", 
        base_path / "randomized_search.html"
    )
