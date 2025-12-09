"""
Real-time predictor for TrendFollower.
Use trained models to generate predictions on live data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from config import TrendFollowerConfig, DEFAULT_CONFIG
from data_loader import aggregate_to_bars
from feature_engine import calculate_features_for_timeframe, get_feature_columns
from models import TrendFollowerModels


@dataclass
class TrendSignal:
    """Signal from the trend classifier"""
    timestamp: datetime
    direction: int  # -1, 0, 1
    prob_up: float
    prob_down: float
    prob_neutral: float
    confidence: float  # max probability
    regime: int
    regime_name: str


@dataclass
class EntrySignal:
    """Signal for a potential entry"""
    timestamp: datetime
    direction: int
    bounce_prob: float
    expected_rr: float
    is_pullback_zone: bool
    trend_aligned: bool
    signal_quality: str  # 'A', 'B', 'C'


class TrendFollowerPredictor:
    """
    Real-time predictor using trained models.
    
    Usage:
        predictor = TrendFollowerPredictor()
        predictor.load_models('./models')
        
        # Feed new trades
        predictor.add_trades(new_trades_df)
        
        # Get predictions
        trend_signal = predictor.get_trend_signal()
        entry_signal = predictor.get_entry_signal()
    """
    
    REGIME_NAMES = {
        0: 'ranging',
        1: 'trending_up',
        2: 'trending_down',
        3: 'volatile'
    }
    
    def __init__(self, config: TrendFollowerConfig = DEFAULT_CONFIG):
        self.config = config
        self.models: Optional[TrendFollowerModels] = None
        
        # Trade buffer
        self.trades_buffer: List[pd.DataFrame] = []
        self.max_buffer_size = 100000  # Max trades to keep in memory
        
        # Bar caches
        self.bars_cache: Dict[str, pd.DataFrame] = {}
        self.features_cache: Optional[pd.DataFrame] = None
        
        # Last prediction
        self.last_trend_signal: Optional[TrendSignal] = None
        self.last_entry_signal: Optional[EntrySignal] = None
        
        self.base_tf = config.features.timeframe_names[config.base_timeframe_idx]
        self.feature_cols: List[str] = []
    
    def load_models(self, model_dir: Path):
        """Load trained models"""
        model_dir = Path(model_dir)
        
        self.models = TrendFollowerModels(self.config.model)
        self.models.load_all(model_dir)
        
        # Get feature columns from model
        self.feature_cols = self.models.trend_classifier.feature_names
        
        print(f"Loaded models from {model_dir}")
        print(f"Expected features: {len(self.feature_cols)}")
    
    def add_trades(self, trades: pd.DataFrame):
        """
        Add new trades to the buffer and update predictions.
        
        Args:
            trades: DataFrame with trade data (same format as training)
        """
        self.trades_buffer.append(trades)
        
        # Trim buffer if too large
        total_trades = sum(len(df) for df in self.trades_buffer)
        while total_trades > self.max_buffer_size and len(self.trades_buffer) > 1:
            removed = self.trades_buffer.pop(0)
            total_trades -= len(removed)
        
        # Rebuild bars and features
        self._update_features()
    
    def _update_features(self):
        """Rebuild bars and features from trade buffer"""
        if not self.trades_buffer:
            return
        
        # Combine all trades
        all_trades = pd.concat(self.trades_buffer, ignore_index=True)
        all_trades = all_trades.sort_values(self.config.data.timestamp_col)
        
        # Preprocess
        from data_loader import preprocess_trades
        all_trades = preprocess_trades(all_trades, self.config.data)
        
        # Create bars for each timeframe
        for tf_seconds, tf_name in zip(self.config.features.timeframes, 
                                        self.config.features.timeframe_names):
            bars = aggregate_to_bars(all_trades, tf_seconds, self.config.data)
            self.bars_cache[tf_name] = bars
        
        # Calculate features
        from feature_engine import calculate_multi_timeframe_features
        self.features_cache = calculate_multi_timeframe_features(
            self.bars_cache,
            self.base_tf,
            self.config.features
        )
    
    def get_trend_signal(self) -> Optional[TrendSignal]:
        """
        Get current trend prediction.
        
        Returns:
            TrendSignal with direction and probabilities
        """
        if self.models is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        if self.features_cache is None or len(self.features_cache) == 0:
            return None
        
        # Get latest features
        latest = self.features_cache.iloc[[-1]]
        
        # Prepare features efficiently (avoid fragmentation)
        feature_data = {}
        for col in self.feature_cols:
            if col in latest.columns:
                feature_data[col] = latest[col].fillna(0).values
            else:
                feature_data[col] = [0]
        
        X = pd.DataFrame(feature_data, index=latest.index)
        
        # Predict trend
        trend_pred = self.models.trend_classifier.predict(X)
        
        # Predict regime
        regime_pred = self.models.regime_classifier.predict(X)
        
        # Create signal
        signal = TrendSignal(
            timestamp=datetime.now(),
            direction=int(trend_pred['prediction'][0]),
            prob_up=float(trend_pred['prob_up'][0]),
            prob_down=float(trend_pred['prob_down'][0]),
            prob_neutral=float(trend_pred['prob_neutral'][0]),
            confidence=float(max(trend_pred['prob_up'][0], 
                                trend_pred['prob_down'][0], 
                                trend_pred['prob_neutral'][0])),
            regime=int(regime_pred['regime'][0]),
            regime_name=self.REGIME_NAMES.get(regime_pred['regime'][0], 'unknown')
        )
        
        self.last_trend_signal = signal
        return signal
    
    def get_entry_signal(self) -> Optional[EntrySignal]:
        """
        Get entry quality prediction for current bar.
        
        Returns:
            EntrySignal with bounce probability and quality grade
        """
        if self.models is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        if self.features_cache is None or len(self.features_cache) == 0:
            return None
        
        # Get latest features
        latest = self.features_cache.iloc[[-1]]
        
        # Prepare features efficiently (avoid fragmentation)
        feature_data = {}
        for col in self.feature_cols:
            if col in latest.columns:
                feature_data[col] = latest[col].fillna(0).values
            else:
                feature_data[col] = [0]
        
        X = pd.DataFrame(feature_data, index=latest.index)
        
        # Check if in pullback zone
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        atr_col = f'{self.base_tf}_atr'
        alignment_col = f'{self.base_tf}_ema_alignment'
        
        is_pullback = False
        trend_dir = 0
        
        if all(col in latest.columns for col in [ema_col, atr_col, alignment_col]):
            price = latest['close'].iloc[0]
            ema = latest[ema_col].iloc[0]
            atr = latest[atr_col].iloc[0]
            alignment = latest[alignment_col].iloc[0]
            
            if atr > 0:
                dist = abs(price - ema) / atr
                is_pullback = dist <= self.config.labels.pullback_threshold
                trend_dir = np.sign(alignment) if abs(alignment) > 0.3 else 0
        
        # Predict entry quality
        entry_pred = self.models.entry_model.predict(X)
        
        bounce_prob = float(entry_pred['bounce_prob'][0])
        expected_rr = float(entry_pred.get('expected_rr', [1.0])[0])
        
        # Determine signal quality
        trend_signal = self.get_trend_signal()
        trend_aligned = (trend_signal is not None and 
                        trend_signal.direction == trend_dir and 
                        trend_dir != 0)
        
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
        else:
            quality = 'C'
        
        signal = EntrySignal(
            timestamp=datetime.now(),
            direction=int(trend_dir),
            bounce_prob=bounce_prob,
            expected_rr=expected_rr,
            is_pullback_zone=is_pullback,
            trend_aligned=trend_aligned,
            signal_quality=quality
        )
        
        self.last_entry_signal = signal
        return signal
    
    def get_full_prediction(self) -> Dict:
        """
        Get complete prediction summary.
        
        Returns:
            Dictionary with all predictions and context
        """
        trend = self.get_trend_signal()
        entry = self.get_entry_signal()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'trend': None,
            'entry': None,
            'recommendation': 'WAIT'
        }
        
        if trend:
            result['trend'] = {
                'direction': trend.direction,
                'direction_name': {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}[trend.direction],
                'prob_up': trend.prob_up,
                'prob_down': trend.prob_down,
                'confidence': trend.confidence,
                'regime': trend.regime_name
            }
        
        if entry:
            result['entry'] = {
                'direction': entry.direction,
                'bounce_prob': entry.bounce_prob,
                'expected_rr': entry.expected_rr,
                'is_pullback': entry.is_pullback_zone,
                'trend_aligned': entry.trend_aligned,
                'quality': entry.signal_quality
            }
            
            # Generate recommendation
            if entry.signal_quality == 'A' and entry.direction != 0:
                result['recommendation'] = f"ENTER {'LONG' if entry.direction > 0 else 'SHORT'} (A-grade)"
            elif entry.signal_quality == 'B' and entry.direction != 0:
                result['recommendation'] = f"CONSIDER {'LONG' if entry.direction > 0 else 'SHORT'} (B-grade)"
            else:
                result['recommendation'] = 'WAIT'
        
        return result


def create_live_predictor(model_dir: Path, config: TrendFollowerConfig = DEFAULT_CONFIG) -> TrendFollowerPredictor:
    """
    Create and initialize a live predictor.
    
    Args:
        model_dir: Path to saved models
        config: Configuration
        
    Returns:
        Initialized TrendFollowerPredictor
    """
    predictor = TrendFollowerPredictor(config)
    predictor.load_models(model_dir)
    return predictor


if __name__ == "__main__":
    print("Predictor module loaded successfully")
    
    # Example usage (requires trained models)
    # predictor = create_live_predictor(Path('./models'))
    # predictor.add_trades(new_trades_df)
    # prediction = predictor.get_full_prediction()
    # print(prediction)
