from .config import Settings, load_settings, MODEL_DIR, initialize_config, update_fees, load_fees
from .massive_client import create_massive_client
from .cli import create_parser
from argparse import ArgumentParser
from .typedefs import MassiveParameters
from .fetch import fetch
from .datareader import fetch_contract_spec
from .preprocessing.preprocess import preprocess
from .model import run_model
from .validate import validate_input
from .typedefs import (
    FetchLatestArgs,
    FetchLookbackArgs,
    FetchRangeArgs,
    ModelArgs,
    PreprocessArgs,
    InputArgs,
    TradingFeeUpdates,
    PredictionInterval,
)

def main():
    settings: Settings = load_settings()
    parser: ArgumentParser = create_parser()

    cli_args: dict = vars(settings.defaults) | vars(parser.parse_args())

    reset: bool = cli_args.get("reset", False)
    initialize_config(reset)

    if (reset):
        return

    args: InputArgs = validate_input(cli_args)

    match args:
        case FetchLatestArgs() | FetchRangeArgs() | FetchLookbackArgs():
            massive_parameters: MassiveParameters = {
                # "limit": 100,
                "sort": "window_start.desc",
                "resolution": args.resolution,
                "ticker": args.ticker,
            }

            fetch(
                massive_parameters,
                create_massive_client(settings.massive_api_key),
                args,
            )

        case PreprocessArgs():
            preprocess(args.ticker, args.resolution)

        case ModelArgs():
            hf_token = settings.hf_token
            contract_spec = fetch_contract_spec(args.ticker, create_massive_client(settings.massive_api_key))
            pred_interval: PredictionInterval = PredictionInterval.model_validate(
                {
                    "lower": args.prediction_interval[0],
                    "upper": args.prediction_interval[1],
                }
            )
            run_model(
                    args.ticker, 
                    args.target,
                    args.pred_length, 
                    args.quantiles,
                    pred_interval,
                    contract_spec,
                    hf_token=hf_token, 
                    model_dir=MODEL_DIR/args.ticker, 
                    store_weights=args.store_weights, 
                    eval=args.eval
            )

        case TradingFeeUpdates():
            update_fees(args)
        case _:
            return

if __name__ == "__main__":
    main()
