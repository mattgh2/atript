from .model import (
    TREND,
    MOMENTUM,
    VOLATILITY,
    VOLUME,
    TIME,
    MISSING,
    load_chronos,
    build_training_input,
    build_validation_input,
    TrainTestValidate,
    load_chronos, 
    evaluate,
)
from .model.evaluate import predict_chronos
from .model.backtest import backtest_strategy
from pathlib import Path
from itertools import chain
from transformers import EarlyStoppingCallback
import numpy as np
from .datareader import fetch_contract_spec
import pandas as pd
from .figs import plot_predictions, parameter_performance
from .config import FIGURE_PATH, WEB_PATH
from .web_util import display_figs_on_page, update_resolution_page
from collections.abc import Sequence, Iterable
from .typedefs import ContractSpec, BacktestResults

"""
Start with an all features baseline.
Test: No volume
      no Trend
      no momentum
      no volatilty  
      no calendar/session-time
      no missingness/gap features

      Only core OHLCV / minimal covariates

      Price + time features only
      No trend + momentum
      No volatility  + range-based indicators
      NO missingness + elapsed-time feature
      NO volume derived features together.

      For each group, compare the full set with a minimal set.

"""

def run_experiment(
    name: str,
    covariates: list[str],
    dataset: TrainTestValidate,
    pred_length: int,
    context_length: int,
    quantiles,
    prediction_interval,
    resolution
) -> tuple[Path, Sequence[str]]:
    train_df, test_df, validation_df = dataset

    train_input = build_training_input(train_df, covariates, pred_length=pred_length)
    validation_input, validation_windows = build_validation_input(
        validation_df, covariates, pred_length, context_length
    )

    pipeline = load_chronos()

    pipeline = pipeline.fit(
            inputs=train_input,
            validation_inputs=validation_input,
            prediction_length=pred_length,
            context_length=context_length,
            learning_rate=1e-6,
            num_steps=1000,
            batch_size=256,
            finetune_mode='full',
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            eval_steps=100,
            save_steps=100
    )

    fig_path: Path = FIGURE_PATH / f"abalation/finetuned/{resolution}"
    fig_path.mkdir(parents=True, exist_ok=True)

    predictions: list[pd.DataFrame] = []
    for window in validation_windows:

        context_df = window['context'][['ticker', 'model_timestamp', 'close', *covariates]]
        actual = window['actual']

        pred_df = predict_chronos(
            pipeline,
            context_df,
            pred_length,
            context_length,
            quantiles,
        )

        pred_df = pd.merge(
                left=pred_df,
                right=actual,
                on=['ticker', "model_timestamp"],
                validate="one_to_one",
                how="inner"
        )

        pred_df['origin_close'] = context_df['close'].iloc[-1]
        pred_df['forecast_origin'] = context_df['model_timestamp'].iloc[-1]
        pred_df['horizon'] = np.arange(1, len(pred_df) + 1)

        predictions.append(pred_df)

    if len(predictions) == 0:
        raise ValueError("No predictions.")

    eval_df = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(by="model_timestamp")
        .reset_index(drop=True)
    )

    first_contract = eval_df['ticker'].unique()[0]

    plot_df = eval_df[eval_df['ticker'].eq(first_contract)]

    plot_y_pred = plot_df['predictions'].head(200)
    plot_y_true = plot_df['close'].head(200)
    plot_lower = plot_df[str(prediction_interval.lower)].head(200)
    plot_upper = plot_df[str(prediction_interval.upper)].head(200)

    eval_results = evaluate(eval_df, quantiles, prediction_interval)

    all_trades: list[pd.DataFrame] = []
    contract_summaries: list[BacktestResults] = []
    total_missing = 0

    for ticker, forecast_df in eval_df.groupby('ticker', sort=False):
        market_df = validation_df[validation_df['ticker'].eq(ticker)]

        trades, summary = backtest_strategy(
                forecast_df,
                market_df,
                prediction_interval,
                fetch_contract_spec(str(ticker)),
                pred_length,
        )

        total_missing += summary.missing_count

        all_trades.append(trades)
        contract_summaries.append(summary)

    signal_ledger = pd.concat(all_trades, ignore_index=True)

    allocated_capital: float = 1000.0
    risk_free_per_window: float = 0.0
    returns: pd.Series = signal_ledger['net_pnl'] / allocated_capital
    excess: pd.Series = returns - risk_free_per_window
    std = excess.std(ddof=1)
    net_sharpe_per_window = excess.mean() / std if len(excess) >= 2 and std > 0 else float('nan')

    trade_ledger = signal_ledger[signal_ledger['signal'].ne(0)]

    summary = BacktestResults(
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

    current_fig_path = fig_path / f"{name}-predictions.png"

    plot_predictions(
        plot_y_pred,
        plot_y_true,
        plot_lower,
        plot_upper,
        current_fig_path,
    )

    output: list[str] = list(
        chain(eval_results.to_strings(), ["\n"], summary.to_strings())
    )

    return (current_fig_path, output)


def abalation(
    dataset: TrainTestValidate,
    pred_length,
    context_length,
    quantiles,
    prediction_interval,
    resolution,
):

    features_groups = {
            "Trend": TREND,
            "Volatility": VOLATILITY,
            "Momentum": MOMENTUM,
            "Time": TIME,
            "Missing": MISSING,
            "Volume": VOLUME,
    }

    def covariates_for(*, exclude: frozenset[str] = frozenset()) -> list[str]:
        return list(
            chain.from_iterable(
                values for name, values in features_groups.items() if name not in exclude
            )
        )

    experiments = {
            f"no_{group}": covariates_for(exclude=frozenset({group})) 
            for group in features_groups
    }

    experiments['full_set'] = covariates_for(exclude=frozenset({}))

    figs: list[tuple[Path, Sequence[str]]] = []
    for name, covariates in experiments.items():
        figs.append(
            run_experiment(
                name,
                covariates,
                dataset,
                pred_length,
                context_length,
                quantiles,
                prediction_interval,
                resolution,
            )
        )

    path: Path = WEB_PATH / f"finetuned/abalation"

    update_resolution_page(
        path / "resolutions.html",
        resolution,
        path / f"{resolution}/results.html",
        WEB_PATH / f"finetuned/nav.html",
    )
    (path / resolution).mkdir(exist_ok=True, parents=True)
    display_figs_on_page(figs, WEB_PATH / f"finetuned/abalation/{resolution}/results.html", path / 'resolutions.html')
