"""
Label generation for trend following ML model.
Creates training labels from historical price data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from config import LabelConfig, FeatureConfig


def label_trend_opportunities(
    df: pd.DataFrame,
    config: LabelConfig,
    base_tf: str
) -> pd.DataFrame:
    """
    Label each bar with whether a tradeable trend followed.
    
    Labels:
        1: Strong uptrend followed (price went up significantly)
       -1: Strong downtrend followed (price went down significantly)
        0: No clear trend (choppy, ranging, or small move)
    
    Args:
        df: DataFrame with OHLCV and features
        config: LabelConfig with thresholds
        base_tf: Base timeframe prefix
        
    Returns:
        DataFrame with 'trend_label' column added
    """
    result = df.copy()
    n = len(result)
    window = config.trend_forward_window
    
    # Get ATR column
    atr_col = f'{base_tf}_atr'
    if atr_col not in result.columns:
        raise ValueError(f"ATR column '{atr_col}' not found")
    
    labels = np.zeros(n)
    max_favorable = np.zeros(n)
    max_adverse = np.zeros(n)
    
    for i in range(n - window):
        current_price = result['close'].iloc[i]
        current_atr = result[atr_col].iloc[i]
        
        if current_atr <= 0 or pd.isna(current_atr):
            continue
        
        # Look forward
        future_highs = result['high'].iloc[i+1:i+window+1]
        future_lows = result['low'].iloc[i+1:i+window+1]
        
        # Max move up and down
        max_up = (future_highs.max() - current_price) / current_atr
        max_down = (current_price - future_lows.min()) / current_atr
        
        max_favorable[i] = max(max_up, max_down)
        
        # Strong uptrend: went up significantly without big drawdown first
        if max_up >= config.trend_up_threshold and max_down < config.max_adverse_for_trend:
            labels[i] = 1
            max_adverse[i] = max_down
        # Strong downtrend: went down significantly without big drawup first
        elif max_down >= config.trend_down_threshold and max_up < config.max_adverse_for_trend:
            labels[i] = -1
            max_adverse[i] = max_up
        else:
            labels[i] = 0
            max_adverse[i] = min(max_up, max_down)
    
    result['trend_label'] = labels
    result['trend_max_favorable'] = max_favorable
    result['trend_max_adverse'] = max_adverse
    
    return result


def detect_pullback_zones(
    df: pd.DataFrame,
    config: LabelConfig,
    base_tf: str
) -> pd.Series:
    """
    Detect bars where price is pulling back to a key EMA in a trend.
    
    A pullback zone is where:
    1. Higher timeframe shows a trend (EMA alignment)
    2. Price has pulled back close to an EMA
    
    Args:
        df: DataFrame with features
        config: LabelConfig
        base_tf: Base timeframe prefix
        
    Returns:
        Boolean series indicating pullback zones
    """
    ema_col = f'{base_tf}_ema_{config.pullback_ema}'
    atr_col = f'{base_tf}_atr'
    alignment_col = f'{base_tf}_ema_alignment'
    
    # Check required columns
    for col in [ema_col, atr_col, alignment_col]:
        if col not in df.columns:
            print(f"Warning: Column {col} not found, using fallback")
            return pd.Series(False, index=df.index)
    
    # Distance from EMA in ATR units
    dist_from_ema = abs(df['close'] - df[ema_col]) / df[atr_col].replace(0, np.nan)
    
    # Pullback zone: close to EMA and in a trend
    is_close_to_ema = dist_from_ema <= config.pullback_threshold
    has_trend = abs(df[alignment_col]) > 0.3  # Some trend present
    
    return is_close_to_ema & has_trend


def label_pullback_outcomes(
    df: pd.DataFrame,
    pullback_mask: pd.Series,
    config: LabelConfig,
    base_tf: str
) -> pd.DataFrame:
    """
    Label pullback zones with their outcomes.
    
    For each pullback zone, determine:
    1. Did the trend continue (bounce)?
    2. What was the max favorable excursion?
    3. What was the max adverse excursion?
    
    Args:
        df: DataFrame with features
        pullback_mask: Boolean mask of pullback zones
        config: LabelConfig
        base_tf: Base timeframe prefix
        
    Returns:
        DataFrame with pullback outcome labels
    """
    result = df.copy()
    n = len(result)
    window = config.entry_forward_window
    
    atr_col = f'{base_tf}_atr'
    alignment_col = f'{base_tf}_ema_alignment'
    
    # Initialize label columns
    result['pullback_success'] = np.nan
    result['pullback_mfe'] = np.nan  # Max Favorable Excursion
    result['pullback_mae'] = np.nan  # Max Adverse Excursion
    result['pullback_rr'] = np.nan   # Realized Reward:Risk
    
    pullback_indices = result.index[pullback_mask].tolist()
    
    for idx in pullback_indices:
        i = result.index.get_loc(idx)
        
        if i + window >= n:
            continue
        
        current_price = result['close'].iloc[i]
        current_atr = result[atr_col].iloc[i]
        trend_dir = np.sign(result[alignment_col].iloc[i])  # 1 for up, -1 for down
        
        if current_atr <= 0 or pd.isna(current_atr) or trend_dir == 0:
            continue
        
        # Look forward
        future = result.iloc[i+1:i+window+1]
        
        if trend_dir > 0:  # Uptrend - expecting bounce up
            mfe = (future['high'].max() - current_price) / current_atr
            mae = (current_price - future['low'].min()) / current_atr
        else:  # Downtrend - expecting bounce down
            mfe = (current_price - future['low'].min()) / current_atr
            mae = (future['high'].max() - current_price) / current_atr
        
        # Success: reached target R:R before hitting stop
        target = config.target_rr * config.stop_atr_multiple
        stop = config.stop_atr_multiple
        
        success = (mfe >= target) and (mae < stop)
        rr = mfe / max(mae, 0.1)
        
        result.loc[idx, 'pullback_success'] = int(success)
        result.loc[idx, 'pullback_mfe'] = mfe
        result.loc[idx, 'pullback_mae'] = mae
        result.loc[idx, 'pullback_rr'] = rr
    
    return result


def label_regime(
    df: pd.DataFrame,
    base_tf: str,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Label market regime for each bar.
    
    Regimes:
        0: Ranging/Choppy
        1: Trending Up
        2: Trending Down
        3: High Volatility (no direction)
    
    Args:
        df: DataFrame with features
        base_tf: Base timeframe prefix
        lookback: Bars to look back for regime detection
        
    Returns:
        DataFrame with 'regime' column
    """
    result = df.copy()
    
    adx_col = f'{base_tf}_adx'
    alignment_col = f'{base_tf}_ema_alignment'
    atr_pct_col = f'{base_tf}_atr_percentile'
    
    # Default to ranging
    result['regime'] = 0
    
    # Trending up: ADX > 25 and bullish alignment
    trending_up = (result[adx_col] > 25) & (result[alignment_col] > 0.3)
    result.loc[trending_up, 'regime'] = 1
    
    # Trending down: ADX > 25 and bearish alignment
    trending_down = (result[adx_col] > 25) & (result[alignment_col] < -0.3)
    result.loc[trending_down, 'regime'] = 2
    
    # High volatility: high ATR but no direction
    if atr_pct_col in result.columns:
        high_vol = (result[atr_pct_col] > 0.8) & (abs(result[alignment_col]) < 0.3)
        result.loc[high_vol, 'regime'] = 3
    
    return result


def create_training_dataset(
    df: pd.DataFrame,
    config: LabelConfig,
    feature_config: FeatureConfig,
    base_tf: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create complete training dataset with all labels.
    
    Args:
        df: DataFrame with all features
        config: LabelConfig
        feature_config: FeatureConfig
        base_tf: Base timeframe prefix
        
    Returns:
        Tuple of (labeled DataFrame, list of feature columns)
    """
    print("Generating labels...")
    
    # Label trend opportunities
    print("  Labeling trend opportunities...")
    result = label_trend_opportunities(df, config, base_tf)
    
    # Detect pullback zones
    print("  Detecting pullback zones...")
    pullback_mask = detect_pullback_zones(result, config, base_tf)
    print(f"    Found {pullback_mask.sum():,} pullback zones")
    
    # Label pullback outcomes
    print("  Labeling pullback outcomes...")
    result = label_pullback_outcomes(result, pullback_mask, config, base_tf)
    
    # Label regime
    print("  Labeling regime...")
    result = label_regime(result, base_tf)
    
    # Get feature columns
    from feature_engine import get_feature_columns
    feature_cols = get_feature_columns(result)
    
    # Remove rows with NaN in key columns
    print("  Cleaning data...")
    key_cols = ['trend_label', f'{base_tf}_atr', f'{base_tf}_adx']
    result = result.dropna(subset=[c for c in key_cols if c in result.columns])
    
    print(f"  Final dataset: {len(result):,} samples")
    print(f"  Features: {len(feature_cols)}")
    
    # Label distribution
    if 'trend_label' in result.columns:
        print(f"\n  Trend label distribution:")
        print(f"    Up trends:   {(result['trend_label'] == 1).sum():,}")
        print(f"    Down trends: {(result['trend_label'] == -1).sum():,}")
        print(f"    No trend:    {(result['trend_label'] == 0).sum():,}")
    
    return result, feature_cols


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    print("Label generation module loaded successfully")
    print(f"Trend threshold: {DEFAULT_CONFIG.labels.trend_up_threshold} ATR")
    print(f"Forward window: {DEFAULT_CONFIG.labels.trend_forward_window} bars")
