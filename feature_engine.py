"""
Feature engineering for trend following ML model.
Calculates technical indicators and microstructure features.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from config import FeatureConfig


# =============================================================================
# Technical Indicators
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Dict[str, pd.Series]:
    """
    Average Directional Index with +DI and -DI
    Returns dict with 'adx', 'plus_di', 'minus_di'
    """
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smoothed TR and DM
    atr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()

    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    # DX and ADX
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx_val = dx.rolling(window=period).mean()

    return {
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0
) -> Dict[str, pd.Series]:
    """Bollinger Bands"""
    mid = sma(close, period)
    std_dev = close.rolling(window=period).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev

    return {
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": (upper - lower) / mid,
        "bb_position": (close - lower) / (upper - lower).replace(0, np.nan),
    }


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


# =============================================================================
# Structure Detection
# =============================================================================

def detect_swing_highs(high: pd.Series, lookback: int = 5) -> pd.Series:
    """Detect swing high points"""
    swing_high = pd.Series(False, index=high.index)

    for i in range(lookback, len(high) - lookback):
        window = high.iloc[i - lookback : i + lookback + 1]
        if high.iloc[i] == window.max():
            swing_high.iloc[i] = True

    return swing_high


def detect_swing_lows(low: pd.Series, lookback: int = 5) -> pd.Series:
    """Detect swing low points"""
    swing_low = pd.Series(False, index=low.index)

    for i in range(lookback, len(low) - lookback):
        window = low.iloc[i - lookback : i + lookback + 1]
        if low.iloc[i] == window.min():
            swing_low.iloc[i] = True

    return swing_low


def get_recent_swing_high(
    high: pd.Series,
    swing_highs: pd.Series,
    current_idx: int,
    lookback: int = 50,
) -> Optional[float]:
    """Get the most recent swing high value"""
    start_idx = max(0, current_idx - lookback)
    subset = swing_highs.iloc[start_idx:current_idx]
    if subset.any():
        last_swing_idx = subset[subset].index[-1]
        return high.loc[last_swing_idx]
    return None


def get_recent_swing_low(
    low: pd.Series,
    swing_lows: pd.Series,
    current_idx: int,
    lookback: int = 50,
) -> Optional[float]:
    """Get the most recent swing low value"""
    start_idx = max(0, current_idx - lookback)
    subset = swing_lows.iloc[start_idx:current_idx]
    if subset.any():
        last_swing_idx = subset[subset].index[-1]
        return low.loc[last_swing_idx]
    return None


# =============================================================================
# Feature Calculator (single timeframe)
# =============================================================================

def calculate_features_for_timeframe(
    bars: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Calculate all features for a single timeframe.

    Args:
        bars: OHLCV bar DataFrame
        config: FeatureConfig

    Returns:
        DataFrame with all features added
    """
    df = bars.copy()

    # --- EMAs ---
    for period in config.ema_periods:
        df[f"ema_{period}"] = ema(df["close"], period)
        df[f"ema_{period}_slope"] = df[f"ema_{period}"].diff()

    # Price vs EMAs (normalized by ATR)
    df["atr"] = atr(df["high"], df["low"], df["close"], config.atr_period)

    for period in config.ema_periods:
        df[f"price_vs_ema_{period}"] = (
            df["close"] - df[f"ema_{period}"]
        ) / df["atr"].replace(0, np.nan)
        df[f"ema_{period}_slope_norm"] = (
            df[f"ema_{period}_slope"] / df["atr"].replace(0, np.nan)
        )

    # --- EMA alignment score ---
    if len(config.ema_periods) >= 3:
        df["ema_alignment"] = 0
        # Check if EMAs are stacked (bullish: short > medium > long)
        ema_cols = [f"ema_{p}" for p in sorted(config.ema_periods)]
        for i in range(len(ema_cols) - 1):
            df["ema_alignment"] += (
                df[ema_cols[i]] > df[ema_cols[i + 1]]
            ).astype(int)

        # Normalize to [-1, 1]
        max_score = len(config.ema_periods) - 1
        df["ema_alignment"] = (df["ema_alignment"] - max_score / 2) / (
            max_score / 2
        )

    # --- RSI ---
    df["rsi"] = rsi(df["close"], config.rsi_period)
    df["rsi_slope"] = df["rsi"].diff(3)  # RSI change over 3 bars

    # --- ADX ---
    adx_vals = adx(df["high"], df["low"], df["close"], config.adx_period)
    df["adx"] = adx_vals["adx"]
    df["plus_di"] = adx_vals["plus_di"]
    df["minus_di"] = adx_vals["minus_di"]
    df["di_diff"] = df["plus_di"] - df["minus_di"]
    df["adx_slope"] = df["adx"].diff(3)

    # --- Bollinger Bands ---
    bb = bollinger_bands(df["close"], config.bb_period, config.bb_std)
    df["bb_width"] = bb["bb_width"]
    df["bb_position"] = bb["bb_position"]

    # --- ATR percentile ---
    df["atr_percentile"] = df["atr"].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )

    # --- Volume features ---
    df["volume_sma"] = sma(df["volume"], config.volume_ma_period)
    df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)
    df["obv"] = obv(df["close"], df["volume"])
    df["obv_slope"] = df["obv"].diff(5) / df["volume_sma"].replace(0, np.nan)

    # --- Structure (bias-free swing distances) ---
    # First detect swings
    df["swing_high"] = detect_swing_highs(df["high"], config.swing_lookback)
    df["swing_low"] = detect_swing_lows(df["low"], config.swing_lookback)

    # Raw (look-ahead) swing prices (mostly NaNs except at swing indices)
    raw_swing_high_price = df["high"].where(df["swing_high"])
    raw_swing_low_price = df["low"].where(df["swing_low"])

    # Apply confirmation lag = swing_lookback
    confirmed_swing_high_price = raw_swing_high_price.shift(config.swing_lookback)
    confirmed_swing_low_price = raw_swing_low_price.shift(config.swing_lookback)

    # Forward-fill to get "most recent confirmed" high/low at each bar
    last_confirmed_high = confirmed_swing_high_price.ffill()
    last_confirmed_low = confirmed_swing_low_price.ffill()

    # Distance from swings in ATR units
    df["dist_from_high"] = (last_confirmed_high - df["close"]) / df[
        "atr"
    ].replace(0, np.nan)
    df["dist_from_low"] = (df["close"] - last_confirmed_low) / df[
        "atr"
    ].replace(0, np.nan)

    # --- Microstructure features (if present in bars) ---
    if "buy_sell_imbalance" in df.columns:
        df["imbalance_ma"] = sma(df["buy_sell_imbalance"], 10)
        df["imbalance_slope"] = df["buy_sell_imbalance"].diff(3)

    if "trade_intensity" in df.columns:
        df["intensity_ma"] = sma(df["trade_intensity"], 20)
        df["intensity_ratio"] = df["trade_intensity"] / df[
            "intensity_ma"
        ].replace(0, np.nan)

    if "avg_trade_size" in df.columns:
        df["size_ma"] = sma(df["avg_trade_size"], 20)
        df["size_ratio"] = df["avg_trade_size"] / df["size_ma"].replace(
            0, np.nan
        )

    # --- Price dynamics ---
    df["returns"] = df["close"].pct_change()
    df["returns_volatility"] = df["returns"].rolling(20).std()
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)

    # --- Candle features ---
    df["body_size"] = (df["close"] - df["open"]).abs() / df["atr"].replace(
        0, np.nan
    )
    df["upper_wick"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / df["atr"].replace(0, np.nan)
    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / df["atr"].replace(0, np.nan)
    df["candle_direction"] = np.sign(df["close"] - df["open"])

    return df


# =============================================================================
# Helper: build partial higher-TF bars from base timeframe
# =============================================================================

def build_partial_htf_from_base(
    base_bars: pd.DataFrame,
    tf_seconds: int,
) -> pd.DataFrame:
    """
    Build partial higher-timeframe candles from base timeframe bars.

    For each base bar i, we aggregate all base bars that fall into the same
    higher-TF bucket up to and including i.

    This simulates what the current higher-TF candle looks like at that time,
    *without* ever using future base bars.

    We aggregate not only OHLCV but also the raw microstructure fields needed
    to reconstruct higher-TF microstructure features:
      - value
      - net_volume
      - trade_count
    and then derive:
      - buy_sell_imbalance
      - vwap
      - trade_intensity
      - avg_trade_size
    """
    # Ensure sorted by time
    df = base_bars.sort_values("bar_time").reset_index(drop=True)

    bar_time = df["bar_time"].to_numpy()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    v = df["volume"].to_numpy()

    # These should exist in base_bars from aggregate_to_bars
    value = df.get("value", pd.Series(0.0, index=df.index)).to_numpy()
    net_volume = df.get("net_volume", pd.Series(0.0, index=df.index)).to_numpy()
    trade_count = df.get("trade_count", pd.Series(0.0, index=df.index)).to_numpy()

    n = len(df)

    # Aggregated (partial) higher-TF fields
    htf_bar_time = np.empty(n, dtype="int64")
    htf_open = np.empty(n, dtype="float32")
    htf_high = np.empty(n, dtype="float32")
    htf_low = np.empty(n, dtype="float32")
    htf_close = np.empty(n, dtype="float32")
    htf_volume = np.empty(n, dtype="float32")
    htf_value = np.empty(n, dtype="float32")
    htf_net_volume = np.empty(n, dtype="float32")
    htf_trade_count = np.empty(n, dtype="float32")

    current_bucket = None
    cur_open = cur_high = cur_low = cur_close = 0.0
    cur_volume = 0.0
    cur_value = 0.0
    cur_net_volume = 0.0
    cur_trade_count = 0.0

    for i in range(n):
        t = bar_time[i]
        bucket = (t // tf_seconds) * tf_seconds

        if current_bucket is None or bucket != current_bucket:
            # New higher-TF bucket starts at this base bar
            current_bucket = bucket
            cur_open = float(o[i])
            cur_high = float(h[i])
            cur_low = float(l[i])
            cur_close = float(c[i])
            cur_volume = float(v[i])
            cur_value = float(value[i])
            cur_net_volume = float(net_volume[i])
            cur_trade_count = float(trade_count[i])
        else:
            # Extend current partial candle with this base bar
            if h[i] > cur_high:
                cur_high = float(h[i])
            if l[i] < cur_low:
                cur_low = float(l[i])
            cur_close = float(c[i])

            cur_volume += float(v[i])
            cur_value += float(value[i])
            cur_net_volume += float(net_volume[i])
            cur_trade_count += float(trade_count[i])

        htf_bar_time[i] = current_bucket
        htf_open[i] = cur_open
        htf_high[i] = cur_high
        htf_low[i] = cur_low
        htf_close[i] = cur_close
        htf_volume[i] = cur_volume
        htf_value[i] = cur_value
        htf_net_volume[i] = cur_net_volume
        htf_trade_count[i] = cur_trade_count

    out = pd.DataFrame(
        {
            "bar_time": htf_bar_time,
            "open": htf_open,
            "high": htf_high,
            "low": htf_low,
            "close": htf_close,
            "volume": htf_volume,
            "value": htf_value,
            "net_volume": htf_net_volume,
            "trade_count": htf_trade_count,
        }
    )
    out["datetime"] = pd.to_datetime(out["bar_time"], unit="s")

    # Rebuild higher-TF microstructure exactly like aggregate_to_bars does
    # buy_volume, sell_volume, buy_sell_imbalance, vwap, trade_intensity, avg_trade_size

    # Avoid division by zero with replace(0, np.nan)
    vol_nonzero = out["volume"].replace(0, np.nan)
    trades_nonzero = out["trade_count"].replace(0, np.nan)

    out["buy_volume"] = (out["volume"] + out["net_volume"]) / 2.0
    out["sell_volume"] = (out["volume"] - out["net_volume"]) / 2.0
    out["buy_sell_imbalance"] = out["net_volume"] / vol_nonzero

    out["vwap"] = out["value"] / vol_nonzero
    out["trade_intensity"] = out["trade_count"] / float(tf_seconds)
    out["avg_trade_size"] = out["volume"] / trades_nonzero

    # Fill any NaNs forward (same as aggregate_to_bars)
    out = out.fillna(method="ffill")

    return out


# =============================================================================
# Multi-timeframe Feature Calculation (bias-free version)
# =============================================================================

def calculate_multi_timeframe_features(
    bars_dict: Dict[str, pd.DataFrame],
    base_tf: str,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Calculate features across all timeframes and merge to base timeframe.

    Bias-free version with BOTH lower and higher TFs:

    - Base timeframe: use its own bars and features (fully closed base candles).
    - Lower TFs (tf_seconds < base_seconds): use their own bars/features and
      align by candle CLOSE TIME to the base decision time.
    - Higher TFs (tf_seconds > base_seconds): build PARTIAL candles from the
      base timeframe (no future info) and compute features on that.

    Result: one row per base bar, where each row is a snapshot of everything
    the model would know at that base bar close.
    """
    if base_tf not in bars_dict:
        raise ValueError(f"Base timeframe {base_tf} not in bars_dict")

    print("Calculating multi-timeframe features (with lower + partial higher TFs)...")

    # --- 1) Base timeframe bars & features ---
    base_bars = bars_dict[base_tf].copy()
    base_bars = base_bars.sort_values("bar_time").reset_index(drop=True)

    # Map timeframe name -> seconds
    tf_name_to_sec = {
        name: sec for name, sec in zip(config.timeframe_names, config.timeframes)
    }
    base_seconds = tf_name_to_sec.get(base_tf)
    if base_seconds is None:
        raise ValueError(
            f"Base timeframe {base_tf} not found in config.timeframe_names"
        )

    print(f"  Base timeframe: {base_tf} ({len(base_bars):,} bars, {base_seconds}s)")

    # Features on base timeframe
    base_feats = calculate_features_for_timeframe(base_bars, config).copy()

    # Decision time = base bar close time
    base_feats["decision_time"] = base_feats["bar_time"] + base_seconds

    # Start result as base features
    result = base_feats.copy()

    # Prefix base non-price columns (keep raw price/time columns as-is)
    base_non_price_cols = [
        c
        for c in result.columns
        if c
        not in [
            "bar_time",
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "decision_time",
        ]
    ]
    base_rename_map = {c: f"{base_tf}_{c}" for c in base_non_price_cols}
    result = result.rename(columns=base_rename_map)

    # Ensure sorted by decision_time for merge_asof
    result = result.sort_values("decision_time").reset_index(drop=True)

    # --- 2) LOWER timeframes (e.g. 1m when base is 5m): align by close time ---
    for tf_name, tf_seconds in zip(config.timeframe_names, config.timeframes):
        if tf_name == base_tf:
            continue

        if tf_seconds >= base_seconds:
            # handled as base or higher TF later
            continue

        if tf_name not in bars_dict:
            continue

        print(f"  Adding LOWER TF features from {tf_name} ({tf_seconds}s)...")

        # Use the lower-TF bars directly (they are fully formed candles)
        lt_bars = bars_dict[tf_name].copy()
        lt_bars = lt_bars.sort_values("bar_time").reset_index(drop=True)

        # Compute features on the lower TF
        lt_feats = calculate_features_for_timeframe(lt_bars, config).copy()

        # Define close_time = bar open + tf_seconds
        lt_feats["close_time"] = lt_feats["bar_time"] + tf_seconds

        # We only merge the non-price columns; we don't want to override base OHLC
        lt_feat_cols = [
            c
            for c in lt_feats.columns
            if c
            not in [
                "bar_time",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
            ]
        ]
        if not lt_feat_cols:
            continue

        # Build a reduced DataFrame with close_time + prefixed features
        lt_merge = lt_feats[["close_time"] + lt_feat_cols].copy()
        lt_merge = lt_merge.sort_values("close_time").reset_index(drop=True)

        # Prefix feature names to avoid collisions
        lt_merge = lt_merge.rename(
            columns={c: f"{tf_name}_{c}" for c in lt_feat_cols}
        )

        # Merge_asof: at each base decision_time, use the latest lower-TF close_time <= decision_time
        result = pd.merge_asof(
            result.sort_values("decision_time"),
            lt_merge.sort_values("close_time"),
            left_on="decision_time",
            right_on="close_time",
            direction="backward",
        )

        # Drop the helper close_time; decision_time stays
        if "close_time" in result.columns:
            result = result.drop(columns=["close_time"])

    # --- 3) HIGHER timeframes (e.g. 15m, 1h, 4h): partial candles from base ---
    for tf_name, tf_seconds in zip(config.timeframe_names, config.timeframes):
        if tf_name == base_tf:
            continue

        if tf_seconds <= base_seconds:
            # base or lower TF already handled
            continue

        print(
            f"  Building partial HIGHER TF bars for {tf_name} "
            f"({tf_seconds}s) from base {base_tf}..."
        )

        # Build partial higher-TF bars aligned with base bars
        htf_bars = build_partial_htf_from_base(base_bars, tf_seconds)

        # Compute features on those partial candles
        htf_feats = calculate_features_for_timeframe(htf_bars, config).copy()

        # Non-price columns only
        htf_feat_cols = [
            c
            for c in htf_feats.columns
            if c
            not in [
                "bar_time",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        if not htf_feat_cols:
            continue

        # Align by position: build_partial_htf_from_base returns one row per base bar
        # in the same time order as base_bars / result.
        htf_feats = htf_feats.reset_index(drop=True)
        result = result.reset_index(drop=True)

        for c in htf_feat_cols:
            result[f"{tf_name}_{c}"] = htf_feats[c].values

    # --- 4) Cross-timeframe features (trend agreement, etc.) ---
    result = calculate_cross_tf_features(result, config)

    # Final ordering by bar_time (optional but nice)
    result = result.sort_values("bar_time").reset_index(drop=True)

    return result



# =============================================================================
# Cross-timeframe features & feature column selection
# =============================================================================

def calculate_cross_tf_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Calculate features that combine multiple timeframes"""

    # Trend agreement across timeframes
    tf_names = config.timeframe_names

    # Count how many timeframes have bullish EMA alignment
    alignment_cols = [
        f"{tf}_ema_alignment"
        for tf in tf_names
        if f"{tf}_ema_alignment" in df.columns
    ]
    if alignment_cols:
        df["tf_trend_agreement"] = df[alignment_cols].mean(axis=1)

    # Count how many timeframes have ADX > 25 (trending)
    adx_cols = [f"{tf}_adx" for tf in tf_names if f"{tf}_adx" in df.columns]
    if adx_cols:
        df["tf_trending_count"] = (df[adx_cols] > 25).sum(axis=1)

    # Average RSI across timeframes
    rsi_cols = [f"{tf}_rsi" for tf in tf_names if f"{tf}_rsi" in df.columns]
    if rsi_cols:
        df["tf_avg_rsi"] = df[rsi_cols].mean(axis=1)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns for model training"""
    exclude = [
        "bar_time",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "swing_high",
        "swing_low",
        "obv",  # Non-features
    ]

    # CRITICAL: Exclude all label columns and leaky features
    # These contain future information and would cause data leakage
    label_columns = [
        # Trend labels (future-looking)
        "trend_label",          # Target for main trend model
        "trend_outcome",        # Future outcome
        "trend_success",        # Did trend work? (future)
        "trend_mfe",            # Max Favorable Excursion (future)
        "trend_mae",            # Max Adverse Excursion (future)
        "trend_rr",             # Realized R:R (future)
        "trend_max_favorable",  # Calculated from future prices
        "trend_max_adverse",    # Calculated from future prices

        # Entry/Pullback labels (future-looking)
        "pullback_label",       # Target for entry model
        "pullback_outcome",     # Future outcome
        "pullback_success",     # Did bounce work? (future)
        "pullback_mfe",         # Max Favorable Excursion (future)
        "pullback_mae",         # Max Adverse Excursion (future)
        "pullback_rr",          # Realized R:R (future)
        "is_pullback_zone",     # Used for labeling

        # Regime labels
        "regime",               # Target for regime classifier
    ]
    exclude.extend(label_columns)

    # Also exclude any columns with these patterns that aren't features
    exclude_patterns = [
        "_count",
        "_time",
        "trdMatchID",
        "symbol",
        "_label",
        "_outcome",
        "_success",
        "_mfe",
        "_mae",
        "_rr",
    ]

    feature_cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(pat in col for pat in exclude_patterns):
            continue
        if df[col].dtype in ["float64", "float32", "int64", "int32"]:
            feature_cols.append(col)

    return feature_cols


if __name__ == "__main__":
    # Simple smoke test (requires your config module & data_loader)
    from config import DEFAULT_CONFIG
    from data_loader import (
        load_trades,
        preprocess_trades,
        create_multi_timeframe_bars,
    )

    print("Feature engine module loaded successfully")
    print(f"EMA periods: {DEFAULT_CONFIG.features.ema_periods}")
    print(f"Timeframes: {DEFAULT_CONFIG.features.timeframe_names}")
