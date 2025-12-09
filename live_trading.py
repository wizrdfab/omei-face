#!/usr/bin/env python3
"""
Standalone Live Paper Trading Test for TrendFollower

This script connects to Bybit via WebSocket, receives real-time trades,
feeds them to the TrendFollower ML model, opens paper positions,
tracks them to completion (stop loss or take profit), and logs results.

The goal is to verify that live performance matches backtest results.

Usage:
    python live_trading.py --model-dir ./models --symbol MONUSDT

Requirements:
    pip install pybit
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep
from typing import Optional, List, Dict
from collections import deque
from dataclasses import dataclass, field, asdict
import threading
import logging
import argparse
import json

# Pybit imports
from pybit.unified_trading import WebSocket

# Our imports
from predictor import TrendFollowerPredictor
from config import TrendFollowerConfig, DEFAULT_CONFIG


# =============================================================================
# Default Parameters - MUST MATCH BACKTEST
# =============================================================================

# These defaults match SimpleBacktester in backtest.py
DEFAULT_PARAMS = {
    'initial_capital': 10000.0,
    'position_size_pct': 0.02,      # 2% risk per trade
    'stop_loss_atr': 1.0,           # Stop in ATR units
    'take_profit_rr': 2.0,          # Target reward:risk ratio
    'min_quality': 'B',             # Minimum signal quality
    'min_trend_prob': 0.5,          # Minimum trend probability
    'min_bounce_prob': 0.5,         # Minimum bounce probability
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PaperPosition:
    """An open paper trading position"""
    entry_time: datetime
    direction: int              # 1 for long, -1 for short
    entry_price: float
    size: float                 # In base currency units
    stop_loss: float
    take_profit: float
    signal_quality: str
    atr_at_entry: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CompletedTrade:
    """A completed paper trade"""
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    size: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    signal_quality: str
    exit_reason: str            # 'stop_loss', 'take_profit'
    duration_seconds: float
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        d['exit_time'] = self.exit_time.isoformat()
        return d


@dataclass
class LiveStats:
    """Live trading statistics"""
    start_time: datetime = None
    trades_received: int = 0
    predictions_made: int = 0
    signals_generated: int = 0
    positions_opened: int = 0
    positions_closed: int = 0
    
    # Performance metrics (updated as trades close)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    
    # By grade
    trades_by_grade: Dict[str, int] = field(default_factory=lambda: {'A': 0, 'B': 0, 'C': 0})
    wins_by_grade: Dict[str, int] = field(default_factory=lambda: {'A': 0, 'B': 0, 'C': 0})


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


# =============================================================================
# Trade Buffer
# =============================================================================

class TradeBuffer:
    """Thread-safe buffer for accumulating trades."""
    
    def __init__(self, max_trades: int = 50000):
        self.max_trades = max_trades
        self.trades: deque = deque(maxlen=max_trades)
        self.lock = threading.Lock()
        self.trade_count = 0
    
    def add_trade(self, trade: dict):
        """Add a single trade from WebSocket message"""
        with self.lock:
            self.trades.append({
                'timestamp': trade['T'] / 1000,
                'symbol': trade['s'],
                'side': trade['S'],
                'size': float(trade['v']),
                'price': float(trade['p']),
                'tickDirection': trade['L'],
                'trdMatchID': trade['i'],
            })
            self.trade_count += 1
    
    def add_trades_batch(self, trades: List[dict]):
        """Add multiple trades from WebSocket message"""
        for trade in trades:
            self.add_trade(trade)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get all trades as DataFrame"""
        with self.lock:
            if not self.trades:
                return pd.DataFrame()
            return pd.DataFrame(list(self.trades))
    
    def get_trade_count(self) -> int:
        return self.trade_count


# =============================================================================
# Live Paper Trader
# =============================================================================

class LivePaperTrader:
    """
    Live paper trading system that matches backtest logic exactly.
    
    - Opens positions based on ML signals (same criteria as backtest)
    - Tracks positions and closes on stop loss or take profit
    - Logs all trades and calculates metrics
    """
    
    def __init__(
        self,
        model_dir: Path,
        symbol: str = 'MONUSDT',
        testnet: bool = False,
        # Trading parameters - MATCH BACKTEST DEFAULTS
        initial_capital: float = DEFAULT_PARAMS['initial_capital'],
        position_size_pct: float = DEFAULT_PARAMS['position_size_pct'],
        stop_loss_atr: float = DEFAULT_PARAMS['stop_loss_atr'],
        take_profit_rr: float = DEFAULT_PARAMS['take_profit_rr'],
        min_quality: str = DEFAULT_PARAMS['min_quality'],
        min_trend_prob: float = DEFAULT_PARAMS['min_trend_prob'],
        min_bounce_prob: float = DEFAULT_PARAMS['min_bounce_prob'],
        # System parameters
        update_interval: float = 5.0,
        warmup_trades: int = 1000,
        log_dir: Path = Path('./live_results'),
    ):
        self.symbol = symbol
        self.testnet = testnet
        
        # Trading parameters (MUST MATCH BACKTEST)
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_rr = take_profit_rr
        self.min_quality = min_quality
        self.min_trend_prob = min_trend_prob
        self.min_bounce_prob = min_bounce_prob
        
        # System parameters
        self.update_interval = update_interval
        self.warmup_trades = warmup_trades
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.trade_buffer = TradeBuffer()
        
        # Initialize ML predictor
        self.logger.info(f"Loading models from {model_dir}...")
        self.config = TrendFollowerConfig()
        self.predictor = TrendFollowerPredictor(self.config)
        self.predictor.load_models(model_dir)
        self.base_tf = self.config.features.timeframe_names[self.config.base_timeframe_idx]
        
        # WebSocket
        self.ws: Optional[WebSocket] = None
        
        # State
        self.running = False
        self.current_price: float = 0.0
        self.current_high: float = 0.0  # Track high since last check
        self.current_low: float = float('inf')  # Track low since last check
        self.position: Optional[PaperPosition] = None
        self.completed_trades: List[CompletedTrade] = []
        self.stats = LiveStats()
        
        # Session log file
        session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trades_file = self.log_dir / f'trades_{session_time}.json'
        self.stats_file = self.log_dir / f'stats_{session_time}.json'
    
    def start(self):
        """Start the live paper trading system"""
        self._print_config()
        
        self.running = True
        self.stats.start_time = datetime.now()
        
        # Connect to WebSocket
        self._connect_websocket()
        
        # Start main loop
        self._main_loop()
    
    def stop(self):
        """Stop the system"""
        self.logger.info("Stopping...")
        self.running = False
    
    def _print_config(self):
        """Print configuration at startup"""
        self.logger.info("=" * 70)
        self.logger.info("LIVE PAPER TRADING - TrendFollower")
        self.logger.info("=" * 70)
        self.logger.info("")
        self.logger.info("TRADING PARAMETERS (matching backtest):")
        self.logger.info(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        self.logger.info(f"  Position Size:      {self.position_size_pct:.1%} of capital")
        self.logger.info(f"  Stop Loss:          {self.stop_loss_atr} ATR")
        self.logger.info(f"  Take Profit:        {self.take_profit_rr}:1 R:R")
        self.logger.info(f"  Min Quality:        {self.min_quality}")
        self.logger.info(f"  Min Trend Prob:     {self.min_trend_prob:.0%}")
        self.logger.info(f"  Min Bounce Prob:    {self.min_bounce_prob:.0%}")
        self.logger.info("")
        self.logger.info("SYSTEM PARAMETERS:")
        self.logger.info(f"  Symbol:             {self.symbol}")
        self.logger.info(f"  Testnet:            {self.testnet}")
        self.logger.info(f"  Update Interval:    {self.update_interval}s")
        self.logger.info(f"  Warmup Trades:      {self.warmup_trades}")
        self.logger.info(f"  Log Directory:      {self.log_dir}")
        self.logger.info("=" * 70)
    
    def _connect_websocket(self):
        """Connect to Bybit WebSocket"""
        self.logger.info(f"Connecting to Bybit {'testnet' if self.testnet else 'mainnet'}...")
        
        self.ws = WebSocket(
            testnet=self.testnet,
            channel_type="linear",
        )
        
        self.ws.trade_stream(
            symbol=self.symbol,
            callback=self._handle_trade_message
        )
        
        self.logger.info(f"Subscribed to publicTrade.{self.symbol}")
    
    def _handle_trade_message(self, message: dict):
        """Callback for WebSocket trade messages"""
        try:
            if 'data' in message:
                trades = message['data']
                self.trade_buffer.add_trades_batch(trades)
                
                # Update price tracking
                for trade in trades:
                    price = float(trade['p'])
                    self.current_price = price
                    self.current_high = max(self.current_high, price)
                    self.current_low = min(self.current_low, price)
                
                self.stats.trades_received = self.trade_buffer.get_trade_count()
        except Exception as e:
            self.logger.error(f"Error handling trade message: {e}")
    
    def _main_loop(self):
        """Main trading loop"""
        self.logger.info(f"Waiting for {self.warmup_trades} trades to warm up...")
        
        last_prediction_time = None
        
        while self.running:
            try:
                trade_count = self.trade_buffer.get_trade_count()
                
                # Warmup phase
                if trade_count < self.warmup_trades:
                    self.logger.info(f"Warming up... {trade_count}/{self.warmup_trades} trades")
                    sleep(self.update_interval)
                    continue
                
                # Check timing
                now = datetime.now()
                if last_prediction_time is not None:
                    elapsed = (now - last_prediction_time).total_seconds()
                    if elapsed < self.update_interval:
                        sleep(0.5)
                        continue
                
                # Run trading logic
                self._trading_tick()
                last_prediction_time = now
                
                # Reset high/low tracking for next interval
                self.current_high = self.current_price
                self.current_low = self.current_price
                
            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                sleep(1)
        
        self._save_results()
        self._print_summary()
    
    def _trading_tick(self):
        """Single trading tick - check exits, check entries"""
        
        # Update predictor
        trades_df = self.trade_buffer.get_dataframe()
        if trades_df.empty:
            return
        
        self.predictor.add_trades(trades_df)
        self.stats.predictions_made += 1
        
        # Get current ATR
        current_atr = self._get_current_atr()
        
        # Step 1: Check if we need to exit current position
        if self.position is not None:
            self._check_exit(current_atr)
        
        # Step 2: Check if we should enter a new position
        if self.position is None:
            self._check_entry(current_atr)
        
        # Log current state
        self._log_state()
    
    def _check_entry(self, current_atr: float):
        """Check if we should enter a position - MATCHES BACKTEST LOGIC"""
        
        # Get predictions
        trend_signal = self.predictor.get_trend_signal()
        entry_signal = self.predictor.get_entry_signal()
        
        if trend_signal is None or entry_signal is None:
            return
        
        # Get model predictions
        trend_dir = trend_signal.direction
        prob_up = trend_signal.prob_up
        prob_down = trend_signal.prob_down
        bounce_prob = entry_signal.bounce_prob
        
        # Determine quality grade (same logic as backtest)
        alignment = self._get_ema_alignment()
        is_pullback = self._is_pullback_zone(current_atr)
        trend_aligned = (trend_dir == np.sign(alignment)) and trend_dir != 0
        
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
        else:
            quality = 'C'
        
        # Check entry criteria (SAME AS BACKTEST)
        quality_ok = ord(quality) <= ord(self.min_quality)
        
        if trend_dir == 1:
            trend_prob_ok = prob_up >= self.min_trend_prob
        elif trend_dir == -1:
            trend_prob_ok = prob_down >= self.min_trend_prob
        else:
            trend_prob_ok = False
        
        bounce_ok = bounce_prob >= self.min_bounce_prob
        
        # Enter trade if all conditions met
        if quality_ok and trend_prob_ok and bounce_ok and trend_dir != 0:
            self._open_position(trend_dir, quality, current_atr)
    
    def _open_position(self, direction: int, quality: str, atr: float):
        """Open a new paper position"""
        
        price = self.current_price
        
        # Calculate stop loss and take profit (SAME AS BACKTEST)
        stop_loss = price - (direction * self.stop_loss_atr * atr)
        take_profit = price + (direction * self.stop_loss_atr * self.take_profit_rr * atr)
        
        # Calculate position size (SAME AS BACKTEST)
        risk_amount = self.capital * self.position_size_pct
        risk_per_unit = abs(price - stop_loss)
        size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        self.position = PaperPosition(
            entry_time=datetime.now(),
            direction=direction,
            entry_price=price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_quality=quality,
            atr_at_entry=atr,
        )
        
        self.stats.positions_opened += 1
        self.stats.signals_generated += 1
        
        dir_name = "LONG" if direction == 1 else "SHORT"
        self.logger.info("=" * 70)
        self.logger.info(f"📈 OPENED {quality}-grade {dir_name} POSITION")
        self.logger.info(f"   Entry:      {price:.6f}")
        self.logger.info(f"   Stop Loss:  {stop_loss:.6f} ({self.stop_loss_atr} ATR)")
        self.logger.info(f"   Take Profit:{take_profit:.6f} ({self.take_profit_rr}:1 R:R)")
        self.logger.info(f"   Size:       {size:.2f} units (${risk_amount:.2f} risk)")
        self.logger.info("=" * 70)
    
    def _check_exit(self, current_atr: float):
        """Check if we should exit current position - MATCHES BACKTEST LOGIC"""
        
        if self.position is None:
            return
        
        direction = self.position.direction
        stop = self.position.stop_loss
        target = self.position.take_profit
        
        exit_reason = None
        exit_price = None
        
        # Use high/low since last tick to simulate intrabar price action
        high = self.current_high
        low = self.current_low
        
        # Check stop loss (SAME AS BACKTEST)
        if direction == 1 and low <= stop:
            exit_reason = 'stop_loss'
            exit_price = stop
        elif direction == -1 and high >= stop:
            exit_reason = 'stop_loss'
            exit_price = stop
        
        # Check take profit (SAME AS BACKTEST)
        elif direction == 1 and high >= target:
            exit_reason = 'take_profit'
            exit_price = target
        elif direction == -1 and low <= target:
            exit_reason = 'take_profit'
            exit_price = target
        
        if exit_reason:
            self._close_position(exit_price, exit_reason)
    
    def _close_position(self, exit_price: float, exit_reason: str):
        """Close current position and record trade"""
        
        pos = self.position
        now = datetime.now()
        
        # Calculate P&L (SAME AS BACKTEST)
        pnl = pos.direction * (exit_price - pos.entry_price) * pos.size
        pnl_percent = pos.direction * (exit_price - pos.entry_price) / pos.entry_price * 100
        duration = (now - pos.entry_time).total_seconds()
        
        # Record completed trade
        trade = CompletedTrade(
            entry_time=pos.entry_time,
            exit_time=now,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_quality=pos.signal_quality,
            exit_reason=exit_reason,
            duration_seconds=duration,
        )
        
        self.completed_trades.append(trade)
        
        # Update capital
        self.capital += pnl
        
        # Update stats
        self.stats.positions_closed += 1
        self.stats.total_trades += 1
        self.stats.total_pnl += pnl
        self.stats.total_pnl_percent = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        if pnl > 0:
            self.stats.winning_trades += 1
            self.stats.wins_by_grade[pos.signal_quality] += 1
        else:
            self.stats.losing_trades += 1
        
        self.stats.trades_by_grade[pos.signal_quality] += 1
        
        # Clear position
        self.position = None
        
        # Log
        result = "✅ WIN" if pnl > 0 else "❌ LOSS"
        dir_name = "LONG" if pos.direction == 1 else "SHORT"
        
        self.logger.info("=" * 70)
        self.logger.info(f"{result} - Closed {dir_name} ({exit_reason})")
        self.logger.info(f"   Entry:    {pos.entry_price:.6f}")
        self.logger.info(f"   Exit:     {exit_price:.6f}")
        self.logger.info(f"   P&L:      ${pnl:+.2f} ({pnl_percent:+.2f}%)")
        self.logger.info(f"   Duration: {duration:.0f}s")
        self.logger.info(f"   Capital:  ${self.capital:,.2f}")
        self.logger.info("-" * 70)
        self._log_running_stats()
        self.logger.info("=" * 70)
        
        # Save trade to file
        self._save_trade(trade)
    
    def _log_state(self):
        """Log current market state"""
        trend = self.predictor.get_trend_signal()
        entry = self.predictor.get_entry_signal()
        
        if trend is None or entry is None:
            return
        
        direction_map = {-1: 'DOWN', 0: 'FLAT', 1: 'UP'}
        pos_str = "IN POSITION" if self.position else "NO POSITION"
        
        self.logger.info(
            f"Price: {self.current_price:.6f} | "
            f"Trend: {direction_map[trend.direction]} ({trend.confidence:.0%}) | "
            f"Entry: {entry.signal_quality} ({entry.bounce_prob:.0%}) | "
            f"{pos_str}"
        )
    
    def _log_running_stats(self):
        """Log running statistics"""
        if self.stats.total_trades == 0:
            return
        
        win_rate = self.stats.winning_trades / self.stats.total_trades
        
        self.logger.info(f"   RUNNING STATS:")
        self.logger.info(f"   Trades: {self.stats.total_trades} | "
                        f"Win Rate: {win_rate:.1%} | "
                        f"Total P&L: ${self.stats.total_pnl:+.2f} ({self.stats.total_pnl_percent:+.2f}%)")
    
    def _get_current_atr(self) -> float:
        """Get current ATR from predictor"""
        if self.predictor.features_cache is None:
            return self.current_price * 0.02  # Fallback
        
        atr_col = f'{self.base_tf}_atr'
        if atr_col in self.predictor.features_cache.columns:
            atr = self.predictor.features_cache[atr_col].iloc[-1]
            if pd.notna(atr) and atr > 0:
                return atr
        
        return self.current_price * 0.02
    
    def _get_ema_alignment(self) -> float:
        """Get EMA alignment from predictor"""
        if self.predictor.features_cache is None:
            return 0
        
        col = f'{self.base_tf}_ema_alignment'
        if col in self.predictor.features_cache.columns:
            val = self.predictor.features_cache[col].iloc[-1]
            return val if pd.notna(val) else 0
        return 0
    
    def _is_pullback_zone(self, atr: float) -> bool:
        """Check if price is in pullback zone"""
        if self.predictor.features_cache is None:
            return False
        
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        if ema_col not in self.predictor.features_cache.columns:
            return False
        
        ema = self.predictor.features_cache[ema_col].iloc[-1]
        if pd.isna(ema) or atr <= 0:
            return False
        
        dist = abs(self.current_price - ema) / atr
        return dist <= self.config.labels.pullback_threshold
    
    def _save_trade(self, trade: CompletedTrade):
        """Save trade to file"""
        with open(self.trades_file, 'a') as f:
            f.write(json.dumps(trade.to_dict()) + '\n')
    
    def _save_results(self):
        """Save final results"""
        results = {
            'session_start': self.stats.start_time.isoformat() if self.stats.start_time else None,
            'session_end': datetime.now().isoformat(),
            'symbol': self.symbol,
            'parameters': {
                'initial_capital': self.initial_capital,
                'position_size_pct': self.position_size_pct,
                'stop_loss_atr': self.stop_loss_atr,
                'take_profit_rr': self.take_profit_rr,
                'min_quality': self.min_quality,
                'min_trend_prob': self.min_trend_prob,
                'min_bounce_prob': self.min_bounce_prob,
            },
            'results': {
                'total_trades': self.stats.total_trades,
                'winning_trades': self.stats.winning_trades,
                'losing_trades': self.stats.losing_trades,
                'win_rate': self.stats.winning_trades / self.stats.total_trades if self.stats.total_trades > 0 else 0,
                'total_pnl': self.stats.total_pnl,
                'total_pnl_percent': self.stats.total_pnl_percent,
                'final_capital': self.capital,
                'trades_by_grade': self.stats.trades_by_grade,
                'wins_by_grade': self.stats.wins_by_grade,
            },
            'trades_file': str(self.trades_file),
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {self.stats_file}")
    
    def _print_summary(self):
        """Print final session summary"""
        duration = datetime.now() - self.stats.start_time if self.stats.start_time else timedelta(0)
        
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Duration:           {duration}")
        self.logger.info(f"Trades Received:    {self.stats.trades_received:,}")
        self.logger.info(f"Predictions Made:   {self.stats.predictions_made}")
        self.logger.info(f"Signals Generated:  {self.stats.signals_generated}")
        self.logger.info("")
        self.logger.info("TRADING RESULTS:")
        self.logger.info(f"  Total Trades:     {self.stats.total_trades}")
        self.logger.info(f"  Winning Trades:   {self.stats.winning_trades}")
        self.logger.info(f"  Losing Trades:    {self.stats.losing_trades}")
        
        if self.stats.total_trades > 0:
            win_rate = self.stats.winning_trades / self.stats.total_trades
            self.logger.info(f"  Win Rate:         {win_rate:.1%}")
        
        self.logger.info(f"  Total P&L:        ${self.stats.total_pnl:+,.2f}")
        self.logger.info(f"  Return:           {self.stats.total_pnl_percent:+.2f}%")
        self.logger.info(f"  Final Capital:    ${self.capital:,.2f}")
        
        if self.stats.total_trades > 0:
            self.logger.info("")
            self.logger.info("BY SIGNAL GRADE:")
            for grade in ['A', 'B', 'C']:
                count = self.stats.trades_by_grade.get(grade, 0)
                wins = self.stats.wins_by_grade.get(grade, 0)
                wr = wins / count if count > 0 else 0
                self.logger.info(f"  Grade {grade}: {count} trades, {wr:.1%} win rate")
        
        self.logger.info("=" * 70)
        self.logger.info(f"Trades log: {self.trades_file}")
        self.logger.info(f"Stats file: {self.stats_file}")
        self.logger.info("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TrendFollower Live Paper Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with backtest-matching defaults
    python live_trading.py --model-dir ./models --symbol MONUSDT
    
    # Use testnet
    python live_trading.py --model-dir ./models --symbol MONUSDT --testnet
    
Note: Default parameters match the backtest exactly:
    - min_quality: B
    - min_trend_prob: 0.5 (50%)
    - min_bounce_prob: 0.5 (50%)
    - stop_loss: 1.0 ATR
    - take_profit: 2.0:1 R:R
        """
    )
    
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--symbol', type=str, default='MONUSDT',
                       help='Trading symbol (default: MONUSDT)')
    parser.add_argument('--testnet', action='store_true',
                       help='Use Bybit testnet')
    
    # Trading parameters (defaults match backtest)
    parser.add_argument('--min-quality', type=str, default=DEFAULT_PARAMS['min_quality'],
                       choices=['A', 'B', 'C'],
                       help=f"Minimum signal quality (default: {DEFAULT_PARAMS['min_quality']})")
    parser.add_argument('--min-trend-prob', type=float, default=DEFAULT_PARAMS['min_trend_prob'],
                       help=f"Minimum trend probability (default: {DEFAULT_PARAMS['min_trend_prob']})")
    parser.add_argument('--min-bounce-prob', type=float, default=DEFAULT_PARAMS['min_bounce_prob'],
                       help=f"Minimum bounce probability (default: {DEFAULT_PARAMS['min_bounce_prob']})")
    parser.add_argument('--stop-loss-atr', type=float, default=DEFAULT_PARAMS['stop_loss_atr'],
                       help=f"Stop loss in ATR (default: {DEFAULT_PARAMS['stop_loss_atr']})")
    parser.add_argument('--take-profit-rr', type=float, default=DEFAULT_PARAMS['take_profit_rr'],
                       help=f"Take profit R:R (default: {DEFAULT_PARAMS['take_profit_rr']})")
    
    # System parameters
    parser.add_argument('--update-interval', type=float, default=5.0,
                       help='Seconds between predictions (default: 5.0)')
    parser.add_argument('--warmup-trades', type=int, default=1000,
                       help='Trades before starting (default: 1000)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)
    
    # Check model directory
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return
    
    # Create and start trader
    trader = LivePaperTrader(
        model_dir=model_dir,
        symbol=args.symbol,
        testnet=args.testnet,
        min_quality=args.min_quality,
        min_trend_prob=args.min_trend_prob,
        min_bounce_prob=args.min_bounce_prob,
        stop_loss_atr=args.stop_loss_atr,
        take_profit_rr=args.take_profit_rr,
        update_interval=args.update_interval,
        warmup_trades=args.warmup_trades,
    )
    
    try:
        trader.start()
    except KeyboardInterrupt:
        trader.stop()


if __name__ == "__main__":
    main()
