#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Trading v4 Fixed - TrendFollower

Fixes in this version:
- Better error handling for corrupted saved trades
- Shows data freshness (gap between data and now)
- Clarifies "extra" features in logging
- More verbose data status output
"""
import os
import sys
import time
import json
import gzip
import argparse
import logging
import threading
import gc
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from decimal import Decimal, ROUND_DOWN, ROUND_UP

import pandas as pd
import numpy as np

try:
    from pybit.unified_trading import HTTP, WebSocket
except ImportError:
    raise ImportError("pybit required: pip install pybit")

from config import TrendFollowerConfig
from data_loader import preprocess_trades, create_multi_timeframe_bars
from feature_engine import calculate_multi_timeframe_features, get_feature_columns
from models import TrendFollowerModels

# =============================================================================
# Constants
# =============================================================================
MIN_BARS_5M = 200
MIN_BARS_1H = 50
MIN_BARS_4H = 20

DEFAULT_PARAMS = {
    'position_size_pct': 0.02,
    'stop_loss_atr': 1.0,
    'take_profit_rr': 2.0,
    'min_quality': 'B',
    'min_trend_prob': 0.5,
    'min_bounce_prob': 0.5,
    'leverage': 1,
}

# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class LivePosition:
    symbol: str
    entry_time: datetime
    direction: int
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    signal_quality: str
    atr_at_entry: float
    trend_prob: float = 0.0
    bounce_prob: float = 0.0
    is_pullback: bool = False
    trend_aligned: bool = False

@dataclass
class TradingStats:
    start_time: datetime = None
    historical_trades: int = 0
    realtime_trades: int = 0
    bars_5m: int = 0
    bars_1h: int = 0
    bars_4h: int = 0
    signals_checked: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    quality_A: int = 0
    quality_B: int = 0
    quality_C: int = 0
    trend_up: int = 0
    trend_down: int = 0
    trend_neutral: int = 0

# =============================================================================
# Logging
# =============================================================================
class UTF8FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)

def setup_logging(log_dir: Path, session_id: str) -> logging.Logger:
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f'live_trading_{session_id}.log'
    
    file_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S')
    console_fmt = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
    
    fh = UTF8FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)
    
    logger = logging.getLogger('LiveTrading')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# =============================================================================
# Trade Data Manager with Better Error Handling
# =============================================================================
class TradeDataManager:
    def __init__(self, config: TrendFollowerConfig, logger: logging.Logger,
                 max_trades: int = 8000000, save_dir: Path = None):
        self.config = config
        self.logger = logger
        self.max_trades = max_trades
        self.save_dir = save_dir or Path('./live_trading_logs')
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.trades = deque(maxlen=max_trades)
        self.lock = threading.Lock()
        self.historical_count = 0
        self.realtime_count = 0
        self.last_price = 0.0
        self.high_since_reset = 0.0
        self.low_since_reset = float('inf')
        self.data_config = config.data
        
        # Track data timestamps
        self.oldest_timestamp = None
        self.newest_timestamp = None
        
        self.last_save_time = time.time()
        self.save_interval = 300
    
    def load_from_csv(self, data_dir: Path, days: int = 7) -> int:
        """Load from backtest-format CSV files"""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            self.logger.warning(f"Data directory not found: {data_dir}")
            return 0
        
        csv_files = sorted(data_dir.glob('*.csv'))
        if not csv_files:
            self.logger.warning(f"No CSV files found in {data_dir}")
            return 0
        
        self.logger.info(f"Found {len(csv_files)} CSV files")
        
        # Show available date range
        self.logger.info(f"Available files: {csv_files[0].name} to {csv_files[-1].name}")
        
        files_to_load = csv_files[-min(days + 1, len(csv_files)):]
        self.logger.info(f"Loading {len(files_to_load)} most recent files...")
        
        all_trades = []
        for file in files_to_load:
            try:
                self.logger.info(f"  Loading {file.name}...")
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    all_trades.append({
                        'timestamp': float(row[self.data_config.timestamp_col]),
                        'symbol': row.get('symbol', 'UNKNOWN'),
                        'side': row[self.data_config.side_col],
                        'size': float(row[self.data_config.size_col]),
                        'price': float(row[self.data_config.price_col]),
                        'tickDirection': row.get(self.data_config.tick_direction_col, 'Unknown'),
                    })
                self.logger.info(f"    {len(df):,} trades loaded")
                del df
                gc.collect()
            except Exception as e:
                self.logger.error(f"Error loading {file.name}: {e}")
        
        count = self._add_trades(all_trades, is_historical=True)
        self._log_data_freshness()
        return count
    
    def load_saved_trades(self, symbol: str) -> int:
        """Load previously saved realtime trades with error recovery"""
        pattern = f'saved_trades_{symbol}_*.json.gz'
        files = sorted(self.save_dir.glob(pattern))
        if not files:
            self.logger.info("No saved trades found")
            return 0
        
        # Try files from newest to oldest until one works
        for filepath in reversed(files):
            self.logger.info(f"Trying to load {filepath.name}...")
            try:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    trades = json.load(f)
                
                if trades:
                    count = self._add_trades(trades, is_historical=True)
                    self.logger.info(f"Successfully loaded {count:,} trades from {filepath.name}")
                    return count
                    
            except (gzip.BadGzipFile, json.JSONDecodeError, EOFError) as e:
                self.logger.warning(f"Corrupted file {filepath.name}: {e}")
                # Try to delete corrupted file
                try:
                    filepath.unlink()
                    self.logger.info(f"Deleted corrupted file: {filepath.name}")
                except:
                    pass
                continue
            except Exception as e:
                self.logger.error(f"Error loading {filepath.name}: {e}")
                continue
        
        self.logger.warning("No valid saved trades found")
        return 0
    
    def _add_trades(self, trades: List[dict], is_historical: bool = False) -> int:
        if not trades:
            return 0
        
        trades.sort(key=lambda x: x['timestamp'])
        
        with self.lock:
            for trade in trades:
                self.trades.append(trade)
            
            if is_historical:
                self.historical_count = len(self.trades)
            
            if self.trades:
                self.oldest_timestamp = self.trades[0]['timestamp']
                self.newest_timestamp = self.trades[-1]['timestamp']
                self.last_price = self.trades[-1]['price']
                self.high_since_reset = self.last_price
                self.low_since_reset = self.last_price
        
        return len(trades)
    
    def _log_data_freshness(self):
        """Log how fresh the data is"""
        if self.oldest_timestamp is None or self.newest_timestamp is None:
            return
        
        oldest_dt = datetime.fromtimestamp(self.oldest_timestamp)
        newest_dt = datetime.fromtimestamp(self.newest_timestamp)
        now = datetime.now()
        
        gap = now - newest_dt
        gap_hours = gap.total_seconds() / 3600
        
        self.logger.info("="*60)
        self.logger.info("DATA STATUS")
        self.logger.info(f"  Total trades: {len(self.trades):,}")
        self.logger.info(f"  Oldest: {oldest_dt}")
        self.logger.info(f"  Newest: {newest_dt}")
        self.logger.info(f"  Current time: {now}")
        self.logger.info(f"  DATA GAP: {gap_hours:.1f} hours ({gap})")
        
        if gap_hours > 1:
            self.logger.warning(f"  ⚠️  Data is {gap_hours:.1f} hours old! WebSocket will fill the gap.")
        else:
            self.logger.info(f"  ✓ Data is fresh (gap < 1 hour)")
        self.logger.info("="*60)
    
    def add_realtime_trade(self, trade: dict):
        with self.lock:
            timestamp = trade['T'] / 1000
            price = float(trade['p'])
            
            self.trades.append({
                'timestamp': timestamp,
                'symbol': trade['s'],
                'side': trade['S'],
                'size': float(trade['v']),
                'price': price,
                'tickDirection': trade.get('L', 'Unknown'),
            })
            self.realtime_count += 1
            self.newest_timestamp = timestamp
            self.last_price = price
            self.high_since_reset = max(self.high_since_reset, price)
            self.low_since_reset = min(self.low_since_reset, price)
        
    
    def add_batch(self, trades: List[dict]):
        for t in trades:
            self.add_realtime_trade(t)
    
    def save_trades(self, symbol: str = 'UNKNOWN'):
        """Save trades with atomic write to prevent corruption"""
        with self.lock:
            if not self.trades:
                return
            trades_list = list(self.trades)
            if trades_list:
                symbol = trades_list[0].get('symbol', symbol)
        
        filename = f'saved_trades_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.json.gz'
        filepath = self.save_dir / filename
        temp_path = self.save_dir / f'.tmp_{filename}'
        
        try:
            # Write to temp file first
            with gzip.open(temp_path, 'wt', encoding='utf-8') as f:
                json.dump(trades_list, f)
            
            # Atomic rename
            temp_path.rename(filepath)
            
            self.logger.info(f"Saved {len(trades_list):,} trades to {filename}")
            self.last_save_time = time.time()
            
            # Clean up old save files (keep last 3)
            self._cleanup_old_saves(symbol)
            
        except Exception as e:
            self.logger.error(f"Save error: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def _cleanup_old_saves(self, symbol: str, keep: int = 3):
        """Remove old save files, keeping only the most recent ones"""
        pattern = f'saved_trades_{symbol}_*.json.gz'
        files = sorted(self.save_dir.glob(pattern))
        
        if len(files) > keep:
            for old_file in files[:-keep]:
                try:
                    old_file.unlink()
                    self.logger.debug(f"Cleaned up old save: {old_file.name}")
                except:
                    pass
    
    def get_dataframe(self) -> pd.DataFrame:
        with self.lock:
            if not self.trades:
                return pd.DataFrame()
            return pd.DataFrame(list(self.trades))
    
    def get_price_range(self) -> Tuple[float, float, float]:
        with self.lock:
            return self.last_price, self.high_since_reset, self.low_since_reset
    
    def reset_price_range(self):
        with self.lock:
            self.high_since_reset = self.last_price
            self.low_since_reset = self.last_price
    
    def get_data_gap_hours(self) -> float:
        """Get hours between newest data and now"""
        if self.newest_timestamp is None:
            return float('inf')
        newest_dt = datetime.fromtimestamp(self.newest_timestamp)
        gap = datetime.now() - newest_dt
        return gap.total_seconds() / 3600


# =============================================================================
# Feature Calculator
# =============================================================================
class FeatureCalculator:
    """Uses EXACT same functions as backtest"""
    def __init__(self, config: TrendFollowerConfig, logger: logging.Logger, log_dir: Path = None):
        self.config = config
        self.logger = logger
        self.log_dir = log_dir or Path('./live_trading_logs')
        
        self.timeframes = config.features.timeframes
        self.timeframe_names = config.features.timeframe_names
        self.base_tf = config.features.timeframe_names[config.base_timeframe_idx]
        
        self.features_cache: Optional[pd.DataFrame] = None
        self.bars_cache: Dict[str, pd.DataFrame] = {}
        self.all_feature_cols: List[str] = []
        
        self.last_feature_log = 0
        self.feature_log_interval = 300
    
    def update(self, trades_df: pd.DataFrame) -> Tuple[bool, Dict[str, int]]:
        if trades_df.empty:
            return False, {}
        
        try:
            # Exact backtest pipeline
            preprocessed = preprocess_trades(trades_df, self.config.data)
            
            self.bars_cache = create_multi_timeframe_bars(
                preprocessed,
                self.timeframes,
                self.timeframe_names,
                self.config.data
            )
            
            bar_counts = {tf: len(bars) for tf, bars in self.bars_cache.items()}
            
            if bar_counts.get('5m', 0) < MIN_BARS_5M:
                return False, bar_counts
            
            self.features_cache = calculate_multi_timeframe_features(
                self.bars_cache,
                self.base_tf,
                self.config.features
            )
            
            self.all_feature_cols = get_feature_columns(self.features_cache)
            
            return True, bar_counts
            
        except Exception as e:
            self.logger.error(f"Feature calculation error: {e}")
            import traceback
            traceback.print_exc()
            return False, {}
    
    def get_latest_features(self) -> Optional[pd.DataFrame]:
        if self.features_cache is None or len(self.features_cache) == 0:
            return None
        return self.features_cache.iloc[[-1]]
    
    def get_latest_bar(self) -> Optional[pd.Series]:
        if self.features_cache is None or len(self.features_cache) == 0:
            return None
        return self.features_cache.iloc[-1]
    
    def get_atr(self) -> float:
        if self.features_cache is None:
            return 0.0
        atr_col = f'{self.base_tf}_atr'
        if atr_col in self.features_cache.columns:
            atr = self.features_cache[atr_col].iloc[-1]
            if pd.notna(atr) and atr > 0:
                return float(atr)
        return self.features_cache['close'].iloc[-1] * 0.02
    
    def get_ema_alignment(self) -> float:
        if self.features_cache is None:
            return 0.0
        col = f'{self.base_tf}_ema_alignment'
        if col in self.features_cache.columns:
            val = self.features_cache[col].iloc[-1]
            return float(val) if pd.notna(val) else 0.0
        return 0.0
    
    def is_pullback_zone(self) -> Tuple[bool, float]:
        if self.features_cache is None:
            return False, 999.0
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        atr_col = f'{self.base_tf}_atr'
        if ema_col not in self.features_cache.columns:
            return False, 999.0
        latest = self.features_cache.iloc[-1]
        price = latest['close']
        ema = latest[ema_col]
        atr = latest.get(atr_col, price * 0.02)
        if pd.isna(ema) or atr <= 0:
            return False, 999.0
        dist = abs(price - ema) / atr
        return dist <= self.config.labels.pullback_threshold, float(dist)
    
    def log_feature_status(self, model_features: List[str], force: bool = False):
        now = time.time()
        if not force and (now - self.last_feature_log) < self.feature_log_interval:
            return
        self.last_feature_log = now
        
        if self.features_cache is None:
            self.logger.warning("No features yet")
            return
        
        available = set(self.all_feature_cols)
        needed = set(model_features)
        present = available & needed
        missing = needed - available
        extra = available - needed
        
        self.logger.info("="*70)
        self.logger.info("FEATURE STATUS")
        self.logger.info(f"Model needs: {len(needed)} | We have: {len(available)} | Match: {len(present)} | Missing: {len(missing)}")
        
        # By timeframe
        for tf in self.timeframe_names:
            avail_tf = sum(1 for f in available if f.startswith(f"{tf}_"))
            need_tf = sum(1 for f in needed if f.startswith(f"{tf}_"))
            miss_tf = sum(1 for f in missing if f.startswith(f"{tf}_"))
            bars = len(self.bars_cache.get(tf, []))
            status = "✓ OK" if miss_tf == 0 else f"⚠️ MISSING {miss_tf}"
            self.logger.info(f"  {tf:>3}: {avail_tf:>2}/{need_tf:>2} features | {bars:>4} bars | {status}")
        
        # Cross-TF features
        cross_needed = [f for f in needed if f.startswith('tf_')]
        cross_missing = [f for f in missing if f.startswith('tf_')]
        if cross_needed:
            self.logger.info(f"  Cross-TF: {len(cross_needed) - len(cross_missing)}/{len(cross_needed)} | "
                           f"{'✓ OK' if not cross_missing else f'⚠️ MISSING: {cross_missing}'}")
        
        # Explain "extra" features
        if extra:
            self.logger.info(f"\n  Note: {len(extra)} 'extra' features are intermediate calculations")
            self.logger.info(f"  (e.g., swing_high/low → used to calculate dist_from_high/low)")
            self.logger.info(f"  (e.g., trade_count → used to calculate trade_intensity, avg_trade_size)")
        
        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_needs': len(needed),
            'we_have': len(available),
            'present': len(present),
            'missing': sorted(list(missing)),
            'extra': sorted(list(extra)),
            'extra_explanation': "These are intermediate features used to derive model features",
            'by_timeframe': {tf: {
                'available': sum(1 for f in available if f.startswith(f"{tf}_")),
                'needed': sum(1 for f in needed if f.startswith(f"{tf}_")),
                'bars': len(self.bars_cache.get(tf, []))
            } for tf in self.timeframe_names}
        }
        report_file = self.log_dir / f'feature_report_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"\n  Report saved: {report_file.name}")
        self.logger.info("="*70)

# =============================================================================
# Signal Generator
# =============================================================================
class SignalGenerator:
    def __init__(self, models: TrendFollowerModels, calc: FeatureCalculator,
                 config: TrendFollowerConfig, params: Dict, logger: logging.Logger):
        self.models = models
        self.calc = calc
        self.config = config
        self.params = params
        self.logger = logger
        self.base_tf = config.features.timeframe_names[config.base_timeframe_idx]
    
    def get_signal(self, stats: TradingStats, verbose: bool = True) -> Dict:
        stats.signals_checked += 1
        
        features = self.calc.get_latest_features()
        if features is None:
            return {'direction': 0, 'should_trade': False}
        
        latest = self.calc.get_latest_bar()
        if latest is None:
            return {'direction': 0, 'should_trade': False}
        
        price = float(latest['close'])
        atr = self.calc.get_atr()
        
        # Build feature matrix
        model_features = self.models.trend_classifier.feature_names
        feature_data = {}
        missing = 0
        
        for col in model_features:
            if col in features.columns:
                val = features[col].iloc[0]
                feature_data[col] = [0.0 if pd.isna(val) else float(val)]
            else:
                feature_data[col] = [0.0]
                missing += 1
        
        X = pd.DataFrame(feature_data)
        
        # Get predictions
        trend_pred = self.models.trend_classifier.predict(X)
        entry_pred = self.models.entry_model.predict(X)
        
        trend_dir = int(trend_pred['prediction'][0])
        prob_up = float(trend_pred['prob_up'][0])
        prob_down = float(trend_pred['prob_down'][0])
        bounce_prob = float(entry_pred['bounce_prob'][0])
        
        # Track stats
        if trend_dir == 1: stats.trend_up += 1
        elif trend_dir == -1: stats.trend_down += 1
        else: stats.trend_neutral += 1
        
        alignment = self.calc.get_ema_alignment()
        is_pullback, dist_ema = self.calc.is_pullback_zone()
        trend_aligned = (trend_dir == np.sign(alignment)) and trend_dir != 0
        trend_prob = prob_up if trend_dir == 1 else (prob_down if trend_dir == -1 else max(prob_up, prob_down))
        
        # Quality grading
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
            stats.quality_A += 1
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
            stats.quality_B += 1
        else:
            quality = 'C'
            stats.quality_C += 1
        
        # Entry criteria
        quality_ok = ord(quality) <= ord(self.params['min_quality'])
        trend_prob_ok = (prob_up if trend_dir == 1 else prob_down) >= self.params['min_trend_prob'] if trend_dir != 0 else False
        bounce_ok = bounce_prob >= self.params['min_bounce_prob']
        
        should_trade = quality_ok and trend_prob_ok and bounce_ok and trend_dir != 0
        
        # Get additional info
        rsi = features[f'{self.base_tf}_rsi'].iloc[0] if f'{self.base_tf}_rsi' in features.columns else 0
        adx = features[f'{self.base_tf}_adx'].iloc[0] if f'{self.base_tf}_adx' in features.columns else 0
        tf_agree = features['tf_trend_agreement'].iloc[0] if 'tf_trend_agreement' in features.columns else 0
        
        # Verbose output
        if verbose:
            trend_str = "▲ UP" if trend_dir == 1 else ("▼ DOWN" if trend_dir == -1 else "● NEUTRAL")
            
            self.logger.info("-"*70)
            self.logger.info(f"SIGNAL CHECK #{stats.signals_checked}")
            self.logger.info(f"  Price: {price:.6f} | ATR: {atr:.6f}")
            self.logger.info(f"  Trend: {trend_str} | Prob UP: {prob_up:.1%} | Prob DOWN: {prob_down:.1%}")
            self.logger.info(f"  Bounce Prob: {bounce_prob:.1%} | Quality: {quality}")
            self.logger.info(f"  RSI: {rsi:.1f} | ADX: {adx:.1f} | TF Agreement: {tf_agree:.2f}")
            self.logger.info(f"  Pullback: {'YES' if is_pullback else 'NO'} (dist: {dist_ema:.2f} ATR) | "
                           f"Aligned: {'YES' if trend_aligned else 'NO'}")
            
            if should_trade:
                self.logger.info(f"  >>> ENTRY SIGNAL! <<<")
            else:
                reasons = []
                if trend_dir == 0: reasons.append("no_trend")
                if not quality_ok: reasons.append(f"quality({quality}>{self.params['min_quality']})")
                if not trend_prob_ok and trend_dir != 0: reasons.append(f"trend_prob({trend_prob:.0%}<{self.params['min_trend_prob']:.0%})")
                if not bounce_ok: reasons.append(f"bounce({bounce_prob:.0%}<{self.params['min_bounce_prob']:.0%})")
                self.logger.info(f"  No entry: {', '.join(reasons)}")
        
        return {
            'direction': trend_dir, 'quality': quality, 'trend_prob': trend_prob,
            'prob_up': prob_up, 'prob_down': prob_down, 'bounce_prob': bounce_prob,
            'is_pullback': is_pullback, 'trend_aligned': trend_aligned,
            'dist_from_ema': dist_ema, 'atr': atr, 'price': price,
            'should_trade': should_trade, 'rsi': rsi, 'adx': adx,
            'tf_agreement': tf_agree, 'missing_features': missing,
        }


# =============================================================================
# Position Manager
# =============================================================================
class PositionManager:
    def __init__(self, client: HTTP, symbol: str, params: Dict, 
                 logger: logging.Logger, testnet: bool = True):
        self.client = client
        self.symbol = symbol
        self.params = params
        self.logger = logger
        self.category = "linear"
        self.min_qty = 0.001
        self.qty_step = 0.001
        self.tick_size = 0.00001
        self.min_notional = 1.0
        self._load_instrument_info()
    
    def _load_instrument_info(self):
        try:
            r = self.client.get_instruments_info(category=self.category, symbol=self.symbol)
            if r['retCode'] == 0 and r['result']['list']:
                info = r['result']['list'][0]
                self.min_qty = float(info.get('lotSizeFilter', {}).get('minOrderQty', 0.001))
                self.qty_step = float(info.get('lotSizeFilter', {}).get('qtyStep', 0.001))
                self.min_notional = float(info.get('lotSizeFilter', {}).get('minNotionalValue', 1.0))
                self.tick_size = float(info.get('priceFilter', {}).get('tickSize', 0.00001))
                self.logger.info(f"Instrument: minQty={self.min_qty}, qtyStep={self.qty_step}, tickSize={self.tick_size}")
        except Exception as e:
            self.logger.warning(f"Instrument info error: {e}")
    
    def _round_qty(self, qty: float) -> float:
        step = Decimal(str(self.qty_step))
        return max(float((Decimal(str(qty)) / step).quantize(Decimal('1'), rounding=ROUND_DOWN) * step), self.min_qty)
    
    def _round_price(self, price: float, direction: str = 'down') -> float:
        tick = Decimal(str(self.tick_size))
        rounding = ROUND_DOWN if direction == 'down' else ROUND_UP
        return float((Decimal(str(price)) / tick).quantize(Decimal('1'), rounding=rounding) * tick)
    
    def get_balance(self) -> float:
        try:
            r = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if r['retCode'] == 0:
                for acc in r.get('result', {}).get('list', []):
                    for coin in acc.get('coin', []):
                        if coin.get('coin') == 'USDT':
                            for f in ['availableToWithdraw', 'walletBalance', 'equity']:
                                v = coin.get(f)
                                if v and str(v).strip():
                                    try: return float(v)
                                    except: continue
        except Exception as e:
            self.logger.error(f"Balance error: {e}")
        return 0.0
    
    def get_position(self) -> Optional[Dict]:
        try:
            r = self.client.get_positions(category=self.category, symbol=self.symbol)
            if r['retCode'] == 0:
                for pos in r['result'].get('list', []):
                    if float(pos.get('size', 0)) > 0:
                        return {'side': pos['side'], 'size': float(pos['size']),
                                'entry_price': float(pos.get('avgPrice', 0) or 0)}
        except Exception as e:
            self.logger.error(f"Position error: {e}")
        return None
    
    def open_position(self, direction: int, price: float, atr: float, signal: Dict) -> Optional[LivePosition]:
        side = "Buy" if direction == 1 else "Sell"
        balance = self.get_balance()
        if balance <= 0:
            self.logger.error("No balance!")
            return None
        
        stop_distance = self.params['stop_loss_atr'] * atr
        risk = balance * self.params['position_size_pct']
        qty = self._round_qty(risk / stop_distance if stop_distance > 0 else 0)
        
        if qty * price < self.min_notional:
            qty = self._round_qty(self.min_notional / price * 1.1)
        
        sl = price - (direction * stop_distance)
        tp = price + (direction * stop_distance * self.params['take_profit_rr'])
        
        if direction == 1:
            sl, tp = self._round_price(sl, 'down'), self._round_price(tp, 'up')
        else:
            sl, tp = self._round_price(sl, 'up'), self._round_price(tp, 'down')
        
        try:
            self.client.set_leverage(category=self.category, symbol=self.symbol,
                                    buyLeverage=str(self.params.get('leverage', 1)),
                                    sellLeverage=str(self.params.get('leverage', 1)))
        except: pass
        
        try:
            self.logger.info(f"OPENING {side}: qty={qty}, SL={sl:.6f}, TP={tp:.6f}")
            r = self.client.place_order(category=self.category, symbol=self.symbol, side=side,
                                       orderType="Market", qty=str(qty),
                                       stopLoss=str(sl), takeProfit=str(tp), positionIdx=0)
            if r['retCode'] == 0:
                return LivePosition(symbol=self.symbol, entry_time=datetime.now(),
                                   direction=direction, entry_price=price, size=qty,
                                   stop_loss=sl, take_profit=tp, signal_quality=signal['quality'],
                                   atr_at_entry=atr, trend_prob=signal['trend_prob'],
                                   bounce_prob=signal['bounce_prob'], is_pullback=signal['is_pullback'],
                                   trend_aligned=signal['trend_aligned'])
            else:
                self.logger.error(f"Order failed: {r.get('retMsg')}")
        except Exception as e:
            self.logger.error(f"Order error: {e}")
        return None

# =============================================================================
# Main System
# =============================================================================
class LiveTradingSystem:
    def __init__(self, model_dir: Path, symbol: str = 'MONUSDT', testnet: bool = True,
                 params: Dict = None, data_dir: Optional[Path] = None,
                 log_dir: Path = Path('./live_trading_logs'),
                 decision_interval: int = 300,
                 days_of_history: int = 7):
        self.symbol = symbol
        self.testnet = testnet
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.data_dir = Path(data_dir) if data_dir else None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.decision_interval = decision_interval
        self.days_of_history = days_of_history
        
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = setup_logging(self.log_dir, self.session_id)
        self.config = TrendFollowerConfig()
        
        self.logger.info(f"Loading models from {model_dir}...")
        self.models = TrendFollowerModels(self.config.model)
        self.models.load_all(model_dir)
        self.logger.info(f"Models loaded - expects {len(self.models.trend_classifier.feature_names)} features")
        
        api_key = os.environ.get('BYBIT_API_KEY')
        api_secret = os.environ.get('BYBIT_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("Set BYBIT_API_KEY and BYBIT_API_SECRET")
        
        self.client = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
        
        self.trade_data = TradeDataManager(self.config, self.logger, save_dir=self.log_dir)
        self.feature_calc = FeatureCalculator(self.config, self.logger, self.log_dir)
        self.signal_gen = SignalGenerator(self.models, self.feature_calc, self.config, self.params, self.logger)
        self.position_mgr = PositionManager(self.client, symbol, self.params, self.logger, testnet)
        
        self.ws: Optional[WebSocket] = None
        self.running = False
        self.position: Optional[LivePosition] = None
        self.stats = TradingStats()
        self.features_ready = False
        self.last_decision_time = 0
    
    def start(self):
        self._print_config()
        if not self.testnet:
            self.logger.warning("!!! LIVE TRADING - REAL MONEY !!!")
            if input("Type 'YES': ") != 'YES':
                return
        
        self.running = True
        self.stats.start_time = datetime.now()
        
        self._load_data()
        
        existing = self.position_mgr.get_position()
        if existing:
            self.logger.info(f"Existing position: {existing['side']} {existing['size']}")
        
        self._connect_websocket()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self._shutdown()
    
    def _print_config(self):
        balance = self.position_mgr.get_balance()
        interval_min = self.decision_interval / 60
        self.logger.info("="*70)
        self.logger.info("LIVE TRADING v4 - TrendFollower")
        self.logger.info("="*70)
        self.logger.info(f"Symbol: {self.symbol} | Mode: {'TESTNET' if self.testnet else 'LIVE'}")
        self.logger.info(f"Balance: ${balance:,.2f} | Risk: {self.params['position_size_pct']:.1%}")
        self.logger.info(f"SL: {self.params['stop_loss_atr']} ATR | TP: {self.params['take_profit_rr']}:1")
        self.logger.info(f"Min Quality: {self.params['min_quality']} | Min Trend Prob: {self.params['min_trend_prob']:.0%}")
        self.logger.info(f"Decision Interval: {interval_min:.0f}m | History: {self.days_of_history} days")
        self.logger.info(f"Timeframes: {self.config.features.timeframe_names}")
        self.logger.info("="*70)
    
    def _load_data(self):
        self.logger.info("\n--- LOADING DATA ---")
        
        # Try saved trades (may be corrupted, that's ok)
        self.trade_data.load_saved_trades(self.symbol)
        
        # Load CSV data
        if self.data_dir:
            self.trade_data.load_from_csv(self.data_dir, self.days_of_history)
        
        self.stats.historical_trades = self.trade_data.historical_count
        
        # Show data gap
        gap = self.trade_data.get_data_gap_hours()
        if gap > 24:
            self.logger.warning(f"⚠️  Data is {gap:.0f} hours old!")
            self.logger.warning(f"   You may want to update your CSV files or wait for WebSocket to fill the gap")
        
        # Initial feature calculation
        if self.trade_data.historical_count > 0:
            self._update_features()
    
    def _update_features(self) -> bool:
        trades_df = self.trade_data.get_dataframe()
        if trades_df.empty:
            return False
        
        success, bar_counts = self.feature_calc.update(trades_df)
        
        self.stats.bars_5m = bar_counts.get('5m', 0)
        self.stats.bars_1h = bar_counts.get('1h', 0)
        self.stats.bars_4h = bar_counts.get('4h', 0)
        
        if success and not self.features_ready:
            self.features_ready = True
            self.logger.info(f"✓ Features READY!")
            self.feature_calc.log_feature_status(self.models.trend_classifier.feature_names, force=True)
        
        return success
    
    def _connect_websocket(self):
        self.logger.info(f"\nConnecting to {'testnet' if self.testnet else 'mainnet'} WebSocket...")
        self.ws = WebSocket(testnet=self.testnet, channel_type="linear")
        self.ws.trade_stream(symbol=self.symbol, callback=self._handle_trade)
        self.logger.info(f"✓ Subscribed to publicTrade.{self.symbol}")
        self.logger.info(f"  Waiting for realtime trades...\n")
    
    def _handle_trade(self, msg: dict):
        if 'data' in msg:
            self.trade_data.add_batch(msg['data'])
            self.stats.realtime_trades = self.trade_data.realtime_count
    
    def _main_loop(self):
        self.logger.info("Entering main trading loop...")
        self.logger.info(f"Decision interval: {self.decision_interval}s ({self.decision_interval/60:.0f}m)")
        
        while self.running:
            try:
                now = time.time()
                
                if now - self.last_decision_time < self.decision_interval:
                    time.sleep(1)
                    continue
                
                self.last_decision_time = now
                self._trading_tick()
                self.trade_data.reset_price_range()
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    def _trading_tick(self):
        success = self._update_features()
        
        if not success:
            gap = self.trade_data.get_data_gap_hours()
            self.logger.info(f"Building... 5m:{self.stats.bars_5m}/{MIN_BARS_5M} | "
                           f"1h:{self.stats.bars_1h}/{MIN_BARS_1H} | 4h:{self.stats.bars_4h}/{MIN_BARS_4H} | "
                           f"RT:{self.stats.realtime_trades:,} | Gap:{gap:.1f}h")
            return
        
        price, high, low = self.trade_data.get_price_range()
        
        if self.position is not None:
            status = self._check_position(high, low)
            if status != "open":
                self._handle_exit(status)
        
        if self.position is None:
            self._check_entry()
    
    def _check_position(self, high: float, low: float) -> str:
        if self.position is None:
            return "none"
        
        pos = self.position_mgr.get_position()
        if pos is None:
            d = self.position.direction
            if d == 1:
                return "stop_loss" if low <= self.position.stop_loss else "take_profit"
            else:
                return "stop_loss" if high >= self.position.stop_loss else "take_profit"
        return "open"
    
    def _handle_exit(self, reason: str):
        pos = self.position
        exit_price = pos.stop_loss if reason == "stop_loss" else pos.take_profit
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size
        pnl_pct = pos.direction * (exit_price - pos.entry_price) / pos.entry_price * 100
        
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl
        if pnl > 0: self.stats.winning_trades += 1
        else: self.stats.losing_trades += 1
        
        result = "✓ WIN" if pnl > 0 else "✗ LOSS"
        self.logger.info("="*70)
        self.logger.info(f"{result} | {reason.upper()}")
        self.logger.info(f"  Entry: {pos.entry_price:.6f} → Exit: {exit_price:.6f}")
        self.logger.info(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        self.logger.info(f"  Stats: {self.stats.total_trades}T | W:{self.stats.winning_trades} L:{self.stats.losing_trades} | "
                        f"Total: ${self.stats.total_pnl:+.2f}")
        self.logger.info("="*70)
        self.position = None
    
    def _check_entry(self):
        signal = self.signal_gen.get_signal(self.stats, verbose=True)
        
        if not signal['should_trade']:
            return
        
        self.logger.info("="*70)
        dir_str = "▲ LONG" if signal['direction'] == 1 else "▼ SHORT"
        self.logger.info(f">>> OPENING {signal['quality']}-grade {dir_str} <<<")
        
        pos = self.position_mgr.open_position(signal['direction'], signal['price'], signal['atr'], signal)
        if pos:
            self.position = pos
            self.logger.info(f"  ✓ Entry: {pos.entry_price:.6f}")
            self.logger.info(f"  SL: {pos.stop_loss:.6f} | TP: {pos.take_profit:.6f}")
        else:
            self.logger.error("  ✗ FAILED to open position!")
        self.logger.info("="*70)
    
    def _shutdown(self):
        self.logger.info("\nShutting down...")
        self.trade_data.save_trades(self.symbol)
        self.feature_calc.log_feature_status(self.models.trend_classifier.feature_names, force=True)
        
        duration = datetime.now() - self.stats.start_time if self.stats.start_time else timedelta(0)
        
        self.logger.info("="*70)
        self.logger.info("SESSION SUMMARY")
        self.logger.info(f"  Duration: {duration}")
        self.logger.info(f"  Historical: {self.stats.historical_trades:,} | Realtime: {self.stats.realtime_trades:,}")
        self.logger.info(f"  Signals: {self.stats.signals_checked}")
        self.logger.info(f"  Trades: {self.stats.total_trades} | W/L: {self.stats.winning_trades}/{self.stats.losing_trades}")
        self.logger.info(f"  P&L: ${self.stats.total_pnl:+,.2f}")
        self.logger.info("="*70)

# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Live Trading v4 - Multi-Timeframe')
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--symbol', type=str, default='MONUSDT')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--testnet', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--min-quality', type=str, default='B', choices=['A','B','C'])
    parser.add_argument('--position-size', type=float, default=0.02)
    parser.add_argument('--leverage', type=int, default=1)
    parser.add_argument('--decision-interval', type=int, default=300,
                       help='Seconds between decisions (default: 300 = 5min)')
    parser.add_argument('--history-days', type=int, default=7)
    parser.add_argument('--log-dir', type=str, default='./live_trading_logs')
    args = parser.parse_args()
    
    if not os.environ.get('BYBIT_API_KEY') or not os.environ.get('BYBIT_API_SECRET'):
        print("Error: Set BYBIT_API_KEY and BYBIT_API_SECRET")
        sys.exit(1)
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: {model_dir} not found")
        sys.exit(1)
    
    system = LiveTradingSystem(
        model_dir=model_dir,
        symbol=args.symbol,
        testnet=not args.live,
        params={'position_size_pct': args.position_size, 'min_quality': args.min_quality, 'leverage': args.leverage},
        data_dir=Path(args.data_dir) if args.data_dir else None,
        log_dir=Path(args.log_dir),
        decision_interval=args.decision_interval,
        days_of_history=args.history_days)
    
    system.start()

if __name__ == "__main__":
    main()
