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


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Average Directional Index with +DI and -DI
    Returns dict with 'adx', 'plus_di', 'minus_di'
    """
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
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
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx_val = dx.rolling(window=period).mean()
    
    return {
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di
    }


def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> Dict[str, pd.Series]:
    """Bollinger Bands"""
    mid = sma(close, period)
    std_dev = close.rolling(window=period).std()
    upper = mid + std * std_dev
    lower = mid - std * std_dev
    
    return {
        'bb_mid': mid,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_width': (upper - lower) / mid,
        'bb_position': (close - lower) / (upper - lower).replace(0, np.nan)
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
        window = high.iloc[i-lookback:i+lookback+1]
        if high.iloc[i] == window.max():
            swing_high.iloc[i] = True
    
    return swing_high


def detect_swing_lows(low: pd.Series, lookback: int = 5) -> pd.Series:
    """Detect swing low points"""
    swing_low = pd.Series(False, index=low.index)
    
    for i in range(lookback, len(low) - lookback):
        window = low.iloc[i-lookback:i+lookback+1]
        if low.iloc[i] == window.min():
            swing_low.iloc[i] = True
    
    return swing_low


def get_recent_swing_high(high: pd.Series, swing_highs: pd.Series, current_idx: int, lookback: int = 50) -> Optional[float]:
    """Get the most recent swing high value"""
    start_idx = max(0, current_idx - lookback)
    subset = swing_highs.iloc[start_idx:current_idx]
    if subset.any():
        last_swing_idx = subset[subset].index[-1]
        return high.loc[last_swing_idx]
    return None


def get_recent_swing_low(low: pd.Series, swing_lows: pd.Series, current_idx: int, lookback: int = 50) -> Optional[float]:
    """Get the most recent swing low value"""
    start_idx = max(0, current_idx - lookback)
    subset = swing_lows.iloc[start_idx:current_idx]
    if subset.any():
        last_swing_idx = subset[subset].index[-1]
        return low.loc[last_swing_idx]
    return None


# =============================================================================
# Feature Calculator
# =============================================================================

def calculate_features_for_timeframe(bars: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
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
        df[f'ema_{period}'] = ema(df['close'], period)
        df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff()
    
    # Price vs EMAs (normalized by ATR)
    df['atr'] = atr(df['high'], df['low'], df['close'], config.atr_period)
    
    for period in config.ema_periods:
        df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df['atr'].replace(0, np.nan)
        df[f'ema_{period}_slope_norm'] = df[f'ema_{period}_slope'] / df['atr'].replace(0, np.nan)
    
    # EMA alignment score
    if len(config.ema_periods) >= 3:
        df['ema_alignment'] = 0
        # Check if EMAs are stacked (bullish: short > medium > long)
        ema_cols = [f'ema_{p}' for p in sorted(config.ema_periods)]
        for i in range(len(ema_cols) - 1):
            df['ema_alignment'] += (df[ema_cols[i]] > df[ema_cols[i+1]]).astype(int)
        # Normalize to [-1, 1] range
        max_score = len(config.ema_periods) - 1
        df['ema_alignment'] = (df['ema_alignment'] - max_score/2) / (max_score/2)
    
    # --- RSI ---
    df['rsi'] = rsi(df['close'], config.rsi_period)
    df['rsi_slope'] = df['rsi'].diff(3)  # RSI change over 3 bars
    
    # --- ADX ---
    adx_vals = adx(df['high'], df['low'], df['close'], config.adx_period)
    df['adx'] = adx_vals['adx']
    df['plus_di'] = adx_vals['plus_di']
    df['minus_di'] = adx_vals['minus_di']
    df['di_diff'] = df['plus_di'] - df['minus_di']
    df['adx_slope'] = df['adx'].diff(3)
    
    # --- Bollinger Bands ---
    bb = bollinger_bands(df['close'], config.bb_period, config.bb_std)
    df['bb_width'] = bb['bb_width']
    df['bb_position'] = bb['bb_position']
    
    # --- ATR percentile ---
    df['atr_percentile'] = df['atr'].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # --- Volume features ---
    df['volume_sma'] = sma(df['volume'], config.volume_ma_period)
    df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
    df['obv'] = obv(df['close'], df['volume'])
    df['obv_slope'] = df['obv'].diff(5) / df['volume_sma'].replace(0, np.nan)
    
    # --- Structure ---
    df['swing_high'] = detect_swing_highs(df['high'], config.swing_lookback)
    df['swing_low'] = detect_swing_lows(df['low'], config.swing_lookback)
    
    # Distance from recent swings
    df['dist_from_high'] = np.nan
    df['dist_from_low'] = np.nan
    
    for i in range(len(df)):
        recent_high = get_recent_swing_high(df['high'], df['swing_high'], i)
        recent_low = get_recent_swing_low(df['low'], df['swing_low'], i)
        
        if recent_high is not None and df['atr'].iloc[i] > 0:
            df.loc[df.index[i], 'dist_from_high'] = (recent_high - df['close'].iloc[i]) / df['atr'].iloc[i]
        if recent_low is not None and df['atr'].iloc[i] > 0:
            df.loc[df.index[i], 'dist_from_low'] = (df['close'].iloc[i] - recent_low) / df['atr'].iloc[i]
    
    # --- Microstructure features (from aggregated data) ---
    if 'buy_sell_imbalance' in df.columns:
        df['imbalance_ma'] = sma(df['buy_sell_imbalance'], 10)
        df['imbalance_slope'] = df['buy_sell_imbalance'].diff(3)
    
    if 'trade_intensity' in df.columns:
        df['intensity_ma'] = sma(df['trade_intensity'], 20)
        df['intensity_ratio'] = df['trade_intensity'] / df['intensity_ma'].replace(0, np.nan)
    
    if 'avg_trade_size' in df.columns:
        df['size_ma'] = sma(df['avg_trade_size'], 20)
        df['size_ratio'] = df['avg_trade_size'] / df['size_ma'].replace(0, np.nan)
    
    # --- Price dynamics ---
    df['returns'] = df['close'].pct_change()
    df['returns_volatility'] = df['returns'].rolling(20).std()
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # --- Candle features ---
    df['body_size'] = abs(df['close'] - df['open']) / df['atr'].replace(0, np.nan)
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['atr'].replace(0, np.nan)
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['atr'].replace(0, np.nan)
    df['candle_direction'] = np.sign(df['close'] - df['open'])
    
    return df


def calculate_multi_timeframe_features(
    bars_dict: Dict[str, pd.DataFrame],
    base_tf: str,
    config: FeatureConfig
) -> pd.DataFrame:
    """
    Calculate features across all timeframes and merge to base timeframe.
    
    Args:
        bars_dict: Dictionary of bar DataFrames by timeframe
        base_tf: Base timeframe to merge features into
        config: FeatureConfig
        
    Returns:
        DataFrame with features from all timeframes
    """
    print(f"Calculating features for each timeframe...")
    
    # Calculate features for each timeframe
    featured_bars = {}
    for tf_name, bars in bars_dict.items():
        print(f"  Processing {tf_name}...")
        featured_bars[tf_name] = calculate_features_for_timeframe(bars, config)
    
    # Start with base timeframe
    result = featured_bars[base_tf].copy()
    
    # Rename base TF columns
    feature_cols = [c for c in result.columns if c not in ['bar_time', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
    rename_map = {c: f'{base_tf}_{c}' for c in feature_cols}
    result = result.rename(columns=rename_map)
    
    # Merge higher timeframes
    for tf_name, bars in featured_bars.items():
        if tf_name == base_tf:
            continue
            
        print(f"  Merging {tf_name} features...")
        
        # Get feature columns
        feature_cols = [c for c in bars.columns if c not in ['bar_time', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Rename columns with timeframe prefix
        tf_features = bars[['bar_time'] + feature_cols].copy()
        tf_features = tf_features.rename(columns={c: f'{tf_name}_{c}' for c in feature_cols})
        
        # Merge using asof join (get most recent higher TF bar)
        result = pd.merge_asof(
            result.sort_values('bar_time'),
            tf_features.sort_values('bar_time'),
            on='bar_time',
            direction='backward'
        )
    
    # Calculate cross-timeframe features
    result = calculate_cross_tf_features(result, config)
    
    return result


def calculate_cross_tf_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Calculate features that combine multiple timeframes"""
    
    # Trend agreement across timeframes
    tf_names = config.timeframe_names
    
    # Count how many timeframes have bullish EMA alignment
    alignment_cols = [f'{tf}_ema_alignment' for tf in tf_names if f'{tf}_ema_alignment' in df.columns]
    if alignment_cols:
        df['tf_trend_agreement'] = df[alignment_cols].mean(axis=1)
    
    # Count how many timeframes have ADX > 25 (trending)
    adx_cols = [f'{tf}_adx' for tf in tf_names if f'{tf}_adx' in df.columns]
    if adx_cols:
        df['tf_trending_count'] = (df[adx_cols] > 25).sum(axis=1)
    
    # Average RSI across timeframes
    rsi_cols = [f'{tf}_rsi' for tf in tf_names if f'{tf}_rsi' in df.columns]
    if rsi_cols:
        df['tf_avg_rsi'] = df[rsi_cols].mean(axis=1)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns for model training"""
    exclude = [
        'bar_time', 'datetime', 'open', 'high', 'low', 'close', 'volume',
        'value', 'swing_high', 'swing_low', 'obv'  # Non-features
    ]
    
    # CRITICAL: Exclude all label columns and leaky features
    # These contain future information and would cause data leakage
    label_columns = [
        # Trend labels (future-looking)
        'trend_label',           # The target variable itself!
        'trend_max_favorable',   # Calculated from future prices
        'trend_max_adverse',     # Calculated from future prices
        
        # Entry/Pullback labels (future-looking)
        'pullback_label',        # Target for entry model
        'pullback_outcome',      # Future outcome
        'pullback_success',      # Did bounce work? (future)
        'pullback_mfe',          # Max Favorable Excursion (future)
        'pullback_mae',          # Max Adverse Excursion (future)
        'pullback_rr',           # Realized R:R (future)
        'is_pullback_zone',      # Used for labeling
        
        # Regime labels
        'regime',                # Target for regime classifier
    ]
    exclude.extend(label_columns)
    
    # Also exclude any columns with these patterns that aren't features
    exclude_patterns = [
        '_count', '_time', 'trdMatchID', 'symbol', 
        '_label', '_outcome', '_success', '_mfe', '_mae', '_rr'
    ]
    
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if any(pat in col for pat in exclude_patterns):
            continue
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            feature_cols.append(col)
    
    return feature_cols


if __name__ == "__main__":
    # Test feature calculation
    from config import DEFAULT_CONFIG
    from data_loader import load_trades, preprocess_trades, create_multi_timeframe_bars
    
    # Would need actual data to test
    print("Feature engine module loaded successfully")
    print(f"EMA periods: {DEFAULT_CONFIG.features.ema_periods}")
    print(f"Timeframes: {DEFAULT_CONFIG.features.timeframe_names}")
