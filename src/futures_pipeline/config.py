from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv 
import os
from .typedefs import TradingFeeUpdates, TradingFees
import tomlkit 
from tomlkit.items import Table
from collections.abc import Mapping
from pydantic import ValidationError


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw" 
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
FORECAST_DATA_DIR: Path = DATA_DIR / "forecasts"
TRAINING_DATA_DIR: Path = DATA_DIR / "training"
MODEL_DIR: Path = PROJECT_ROOT / "models"
CONFIG_FILE: Path = PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class Defaults:
    resolution: str = "15min"
    target: str = "returns"
    pred_length: int = 24
    quantiles: list[float] = field(default_factory=lambda: [0.4,0.5,0.6])
    prediction_interval: tuple[float, float] = (0.4,0.6)
@dataclass
class Settings:
    massive_api_key: str
    hf_token: str
    defaults: Defaults = Defaults()
    request_limit: int  = 100


def load_fees() -> TradingFees:
    with open(CONFIG_FILE, 'r') as f:
        config: tomlkit.TOMLDocument = tomlkit.load(f)
    trading_costs: Mapping = config.get("trading_costs", {})

    try:
        fees: TradingFees = TradingFees.model_validate(trading_costs) 
    except ValidationError as e:
        errors = e.errors()
        missing: list[tuple] = [er.get("loc") for er in errors if er.get("type") == "missing"]
        raise ValueError(f"The following fees are not set: {', '.join(name[0] for name in missing)}") from None
    return fees
"""

"""
def update_fees(Fees: TradingFeeUpdates) -> None:
    if (not CONFIG_FILE.exists()):
        raise RuntimeError("Attempted to call upate_fees() when config.toml does not exist in project root. Run config.initialize_config().")

    with open(CONFIG_FILE, 'r') as f:
        config: tomlkit.TOMLDocument = tomlkit.load(f)

    if "trading_costs" not in config:
        config['trading_costs'] = tomlkit.table()

    table: Table = config['trading_costs']
    for type, amount in Fees.model_dump().items():
        if amount is not None:
            table[type] = amount

    with open(CONFIG_FILE, 'w') as f:
        tomlkit.dump(config, f)

def initialize_config(reset=False) -> None:
    config_file = Path(CONFIG_FILE) 

    if config_file.exists() and not reset: 
        return

    config_file.touch(exist_ok=True)

    doc = tomlkit.document()
    trading_costs = tomlkit.table()

    doc.add("trading_costs", trading_costs) 
    with config_file.open('w') as f:
        tomlkit.dump(doc, f)

def load_settings() -> Settings:
    load_dotenv()

    massive_key = os.getenv("MASSIVE_API_KEY")
    if (not massive_key):
        raise RuntimeError("MASSIVE_API_KEY environment variable is not set.")

    hf_token: str | None = os.getenv("HF_TOKEN")
    if (not hf_token):
        raise RuntimeError("HF_TOKEN environment variable is not set.")

    return Settings(massive_api_key=massive_key, hf_token=hf_token)
