# TREND = ['EMA', 'SMA', 'DMA', 'T3MA']

TREND = ['ema_spread_atr', 'ema_slope_atr', 'signed_efficiency_20']
MOMENTUM = ["RSI", 'FSO', 'SSO', 'ROC', 'CCI']

VOLATILITY = [
    "VR",
    "normalized_atr",
    "percent_b",
    "bollinger_bandwidth",
    "keltner_position",
    "keltner_width",
    "bollinger_squeeze_ratio",
    "bollinger_squeeze",
]

TRADING_ACTIVITY = ["log_volume", 'log_transactions']
PRICE_LOCATION = ['close_to_vwap_atr']
SESSION_TIMING = ['sine_time', 'cosine_time', 'is_cash_session', 'minutes_since_cash_open', 'minutes_until_cash_close']
TIME_GAPS = ['has_time_gap', 'log_elapsed_intervals']
MISSING_FLAGS = ['follows_missing', 'was_missing']
NEW_VOLATILITY = ['normalized_atr', 'VR']
RECENT_RETURNS = ['log_returns_1', 'log_returns_5']
NEW_TREND = ['ema_slope_atr', 'ema_spread_atr', 'signed_efficency_ratio']
CANDLE_STRUCTURE = ['signed_candle_body_atr', 'close_location_within_candle_range']
