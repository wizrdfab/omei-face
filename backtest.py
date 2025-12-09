"""
Simple backtester for TrendFollower strategy.
Evaluates model predictions on historical data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config import TrendFollowerConfig, DEFAULT_CONFIG
from models import TrendFollowerModels
from feature_engine import get_feature_columns
from diagnostic_logger import DiagnosticLogger


@dataclass
class Trade:
    """Record of a single trade"""
    entry_time: datetime
    exit_time: datetime
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_percent: float
    signal_quality: str
    exit_reason: str
    # Additional diagnostic info
    trend_prob: float = 0.0
    bounce_prob: float = 0.0
    is_pullback: bool = False
    trend_aligned: bool = False
    dist_from_ema: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    trades_by_grade: Dict[str, int] = field(default_factory=dict)
    win_rate_by_grade: Dict[str, float] = field(default_factory=dict)


class SimpleBacktester:
    """Simple event-driven backtester for model predictions."""
    
    def __init__(
        self,
        models: TrendFollowerModels,
        config: TrendFollowerConfig = DEFAULT_CONFIG,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.02,
        stop_loss_atr: float = 1.0,
        take_profit_rr: float = 2.0,
        min_quality: str = 'B',
        min_trend_prob: float = 0.5,
        min_bounce_prob: float = 0.5,
        diagnostic_logger: Optional[DiagnosticLogger] = None,
    ):
        self.models = models
        self.config = config
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_rr = take_profit_rr
        self.min_quality = min_quality
        self.min_trend_prob = min_trend_prob
        self.min_bounce_prob = min_bounce_prob
        self.base_tf = config.features.timeframe_names[config.base_timeframe_idx]
        self.diag = diagnostic_logger
        
        self.capital = initial_capital
        self.position: Optional[Dict] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        
        # Diagnostic tracking
        self._signal_stats = {
            'total_bars': 0,
            'signals_checked': 0,
            'trend_up_signals': 0,
            'trend_down_signals': 0,
            'trend_neutral_signals': 0,
            'quality_A': 0,
            'quality_B': 0,
            'quality_C': 0,
            'rejected_quality': 0,
            'rejected_trend_prob': 0,
            'rejected_bounce_prob': 0,
            'accepted_signals': 0,
        }
        
        # Grade analysis tracking
        self._grade_analysis = {
            'A': {'signals': [], 'trades': []},
            'B': {'signals': [], 'trades': []},
            'C': {'signals': [], 'trades': []},
        }
    
    def run(self, data: pd.DataFrame, feature_cols: List[str]) -> BacktestResult:
        """Run backtest on historical data."""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        # Reset signal stats
        self._signal_stats = {k: 0 for k in self._signal_stats}
        self._signal_stats['total_bars'] = len(data)
        
        print(f"Running backtest on {len(data):,} bars...")
        
        if self.diag:
            self.diag.log_section("Backtest Execution")
            self.diag.log_metric("backtest_bars", len(data))
            self.diag.log_metric("backtest_initial_capital", self.initial_capital)
            self.diag.log_metric("backtest_min_quality", self.min_quality)
            self.diag.log_metric("backtest_min_trend_prob", self.min_trend_prob)
            self.diag.log_metric("backtest_min_bounce_prob", self.min_bounce_prob)
            self.diag.log_metric("backtest_stop_loss_atr", self.stop_loss_atr)
            self.diag.log_metric("backtest_take_profit_rr", self.take_profit_rr)
            
            # VERIFICATION: Confirm feature_cols doesn't contain label columns
            label_patterns = ['label', 'success', 'outcome', '_mfe', '_mae', '_rr', 'regime']
            leaky_features = [col for col in feature_cols if any(p in col.lower() for p in label_patterns)]
            if leaky_features:
                self.diag.log_error(f"LEAKAGE DETECTED! Feature columns contain labels: {leaky_features}")
            else:
                self.diag.log_raw("\n  ✓ VERIFIED: No label columns in feature_cols\n")
            
            # Log which columns are being used
            self.diag.log_metric("feature_count", len(feature_cols))
            self.diag.log_raw(f"\n  First 10 feature columns: {feature_cols[:10]}\n")
        
        atr_col = f'{self.base_tf}_atr'
        
        for i in range(len(data) - 1):
            row = data.iloc[i]
            next_row = data.iloc[i + 1]
            
            current_price = row['close']
            current_atr = row[atr_col] if atr_col in row else current_price * 0.02
            
            if self.position is not None:
                self._check_exit(row, next_row, current_atr)
            
            if self.position is None:
                self._check_entry(row, data.iloc[[i]], feature_cols, current_price, current_atr)
            
            if self.position is not None:
                unrealized = self._calculate_unrealized_pnl(current_price)
                self.equity_curve.append(self.capital + unrealized)
            else:
                self.equity_curve.append(self.capital)
        
        if self.position is not None:
            final_price = data.iloc[-1]['close']
            self._close_position(final_price, data.iloc[-1].get('datetime', datetime.now()), 'end_of_data')
        
        return self._calculate_results()
    
    def _check_entry(self, row, features_df, feature_cols, price, atr):
        """Check if we should enter a position."""
        self._signal_stats['signals_checked'] += 1
        
        # Build feature DataFrame efficiently (avoid fragmentation)
        feature_data = {}
        for col in feature_cols:
            if col in features_df.columns:
                feature_data[col] = features_df[col].fillna(0).values
            else:
                feature_data[col] = [0]
        
        X = pd.DataFrame(feature_data, index=features_df.index)
        
        trend_pred = self.models.trend_classifier.predict(X)
        entry_pred = self.models.entry_model.predict(X)
        
        trend_dir = trend_pred['prediction'][0]
        prob_up = trend_pred['prob_up'][0]
        prob_down = trend_pred['prob_down'][0]
        bounce_prob = entry_pred['bounce_prob'][0]
        
        # Track trend direction distribution
        if trend_dir == 1:
            self._signal_stats['trend_up_signals'] += 1
        elif trend_dir == -1:
            self._signal_stats['trend_down_signals'] += 1
        else:
            self._signal_stats['trend_neutral_signals'] += 1
        
        alignment_col = f'{self.base_tf}_ema_alignment'
        alignment = row.get(alignment_col, 0)
        
        ema_col = f'{self.base_tf}_ema_{self.config.labels.pullback_ema}'
        if ema_col in row.index:
            ema = row[ema_col]
            dist_from_ema = abs(price - ema) / atr if atr > 0 else 999
            is_pullback = dist_from_ema <= self.config.labels.pullback_threshold
        else:
            is_pullback = False
            dist_from_ema = 999
        
        trend_aligned = (trend_dir == np.sign(alignment)) and trend_dir != 0
        
        # Calculate trend_prob early for signal tracking
        if trend_dir == 1:
            trend_prob = prob_up
        elif trend_dir == -1:
            trend_prob = prob_down
        else:
            trend_prob = max(prob_up, prob_down)
        
        if bounce_prob > 0.6 and trend_aligned and is_pullback:
            quality = 'A'
        elif bounce_prob > 0.5 and (trend_aligned or is_pullback):
            quality = 'B'
        else:
            quality = 'C'
        
        # Track quality distribution
        self._signal_stats[f'quality_{quality}'] += 1
        
        # Track detailed grade analysis for all signals (not just accepted)
        signal_info = {
            'time': row.get('datetime', None),
            'price': price,
            'direction': trend_dir,
            'trend_prob': trend_prob,
            'bounce_prob': bounce_prob,
            'is_pullback': is_pullback,
            'trend_aligned': trend_aligned,
            'alignment': alignment,
            'dist_from_ema': dist_from_ema,
            'atr': atr,
        }
        self._grade_analysis[quality]['signals'].append(signal_info)
        
        quality_ok = ord(quality) <= ord(self.min_quality)
        
        # Check probability thresholds
        if trend_dir == 1:
            trend_prob_ok = prob_up >= self.min_trend_prob
        elif trend_dir == -1:
            trend_prob_ok = prob_down >= self.min_trend_prob
        else:
            trend_prob_ok = False
        
        bounce_ok = bounce_prob >= self.min_bounce_prob
        
        # Track rejection reasons
        if not quality_ok:
            self._signal_stats['rejected_quality'] += 1
        if not trend_prob_ok and trend_dir != 0:
            self._signal_stats['rejected_trend_prob'] += 1
        if not bounce_ok:
            self._signal_stats['rejected_bounce_prob'] += 1
        
        if quality_ok and trend_prob_ok and bounce_ok and trend_dir != 0:
            self._signal_stats['accepted_signals'] += 1
            
            stop_loss = price - (trend_dir * self.stop_loss_atr * atr)
            take_profit = price + (trend_dir * self.stop_loss_atr * self.take_profit_rr * atr)
            
            risk_amount = self.capital * self.position_size_pct
            risk_per_unit = abs(price - stop_loss)
            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            self.position = {
                'direction': trend_dir,
                'entry_price': price,
                'entry_time': row.get('datetime', datetime.now()),
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quality': quality,
                # Additional diagnostic info
                'trend_prob': trend_prob,
                'bounce_prob': bounce_prob,
                'is_pullback': is_pullback,
                'trend_aligned': trend_aligned,
                'dist_from_ema': dist_from_ema,
            }
    
    def _check_exit(self, row, next_row, atr):
        """Check if we should exit current position."""
        high = next_row['high']
        low = next_row['low']
        
        direction = self.position['direction']
        stop = self.position['stop_loss']
        target = self.position['take_profit']
        
        exit_reason = None
        exit_price = None
        
        if direction == 1 and low <= stop:
            exit_reason = 'stop_loss'
            exit_price = stop
        elif direction == -1 and high >= stop:
            exit_reason = 'stop_loss'
            exit_price = stop
        elif direction == 1 and high >= target:
            exit_reason = 'take_profit'
            exit_price = target
        elif direction == -1 and low <= target:
            exit_reason = 'take_profit'
            exit_price = target
        
        if exit_reason:
            self._close_position(exit_price, next_row.get('datetime', datetime.now()), exit_reason)
    
    def _close_position(self, exit_price, exit_time, exit_reason):
        """Close current position and record trade."""
        direction = self.position['direction']
        entry_price = self.position['entry_price']
        size = self.position['size']
        
        pnl = direction * (exit_price - entry_price) * size
        pnl_percent = direction * (exit_price - entry_price) / entry_price * 100
        
        trade = Trade(
            entry_time=self.position['entry_time'],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_quality=self.position['quality'],
            exit_reason=exit_reason,
            trend_prob=self.position.get('trend_prob', 0.0),
            bounce_prob=self.position.get('bounce_prob', 0.0),
            is_pullback=self.position.get('is_pullback', False),
            trend_aligned=self.position.get('trend_aligned', False),
            dist_from_ema=self.position.get('dist_from_ema', 0.0),
        )
        
        # Track trade in grade analysis
        grade = self.position['quality']
        self._grade_analysis[grade]['trades'].append({
            'pnl': pnl,
            'win': pnl > 0,
            'exit_reason': exit_reason,
            'trend_prob': self.position.get('trend_prob', 0.0),
            'bounce_prob': self.position.get('bounce_prob', 0.0),
            'is_pullback': self.position.get('is_pullback', False),
            'trend_aligned': self.position.get('trend_aligned', False),
            'dist_from_ema': self.position.get('dist_from_ema', 0.0),
            'direction': direction,
        })
        
        self.trades.append(trade)
        self.capital += pnl
        self.position = None
    
    def _calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L for open position."""
        if self.position is None:
            return 0.0
        
        direction = self.position['direction']
        entry_price = self.position['entry_price']
        size = self.position['size']
        
        return direction * (current_price - entry_price) * size
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results and metrics."""
        result = BacktestResult()
        result.trades = self.trades
        result.equity_curve = self.equity_curve
        
        # Log signal statistics to diagnostics
        if self.diag:
            self.diag.log_raw("\n  Signal Statistics:\n")
            for key, value in self._signal_stats.items():
                self.diag.log_metric(f"backtest_{key}", value)
            
            # Calculate percentages for better understanding
            total = self._signal_stats['signals_checked']
            if total > 0:
                self.diag.log_raw(f"\n  Signal Breakdown (of {total} bars checked):\n")
                self.diag.log_raw(f"    Trend UP predictions:    {self._signal_stats['trend_up_signals']} ({self._signal_stats['trend_up_signals']/total:.1%})\n")
                self.diag.log_raw(f"    Trend DOWN predictions:  {self._signal_stats['trend_down_signals']} ({self._signal_stats['trend_down_signals']/total:.1%})\n")
                self.diag.log_raw(f"    Trend NEUTRAL predictions: {self._signal_stats['trend_neutral_signals']} ({self._signal_stats['trend_neutral_signals']/total:.1%})\n")
                self.diag.log_raw(f"\n    Quality A signals: {self._signal_stats['quality_A']}\n")
                self.diag.log_raw(f"    Quality B signals: {self._signal_stats['quality_B']}\n")
                self.diag.log_raw(f"    Quality C signals: {self._signal_stats['quality_C']}\n")
                self.diag.log_raw(f"\n    Rejected (quality):      {self._signal_stats['rejected_quality']}\n")
                self.diag.log_raw(f"    Rejected (trend prob):   {self._signal_stats['rejected_trend_prob']}\n")
                self.diag.log_raw(f"    Rejected (bounce prob):  {self._signal_stats['rejected_bounce_prob']}\n")
                self.diag.log_raw(f"    ACCEPTED signals:        {self._signal_stats['accepted_signals']}\n")
        
        if not self.trades:
            if self.diag:
                self.diag.log_warning("No trades were executed in backtest!")
            return result
        
        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        result.losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
        result.win_rate = result.winning_trades / result.total_trades
        
        result.total_pnl = sum(t.pnl for t in self.trades)
        result.total_pnl_percent = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        
        result.avg_win = np.mean(wins) if wins else 0.0
        result.avg_loss = np.mean(losses) if losses else 0.0
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        result.max_drawdown = abs(min(drawdown))
        result.max_drawdown_percent = result.max_drawdown / max(peak) * 100 if max(peak) > 0 else 0.0
        
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        for grade in ['A', 'B', 'C']:
            grade_trades = [t for t in self.trades if t.signal_quality == grade]
            result.trades_by_grade[grade] = len(grade_trades)
            if grade_trades:
                result.win_rate_by_grade[grade] = sum(1 for t in grade_trades if t.pnl > 0) / len(grade_trades)
        
        # Log detailed trade info to diagnostics
        if self.diag:
            self.diag.log_raw("\n  Trade Details:\n")
            self.diag.log_raw(f"    {'#':<3} {'Entry Time':<20} {'Exit Time':<20} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'P&L':<10} {'Result':<6} {'Grade':<5} {'Exit Reason':<12}\n")
            self.diag.log_raw(f"    {'-'*115}\n")
            
            for i, t in enumerate(self.trades):
                dir_str = "LONG" if t.direction == 1 else "SHORT"
                result_str = "WIN" if t.pnl > 0 else "LOSS"
                
                # Format times
                entry_time_str = t.entry_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.entry_time, 'strftime') else str(t.entry_time)[:16]
                exit_time_str = t.exit_time.strftime('%Y-%m-%d %H:%M') if hasattr(t.exit_time, 'strftime') else str(t.exit_time)[:16]
                
                self.diag.log_raw(f"    {i+1:<3} {entry_time_str:<20} {exit_time_str:<20} {dir_str:<5} {t.entry_price:<10.6f} {t.exit_price:<10.6f} ${t.pnl:<9.2f} {result_str:<6} {t.signal_quality:<5} {t.exit_reason:<12}\n")
            
            # Additional trade statistics
            self.diag.log_raw(f"\n  Trade Probabilities:\n")
            self.diag.log_raw(f"    {'#':<3} {'TrendP':<8} {'BounceP':<9} {'Pullback':<10} {'Aligned':<10}\n")
            self.diag.log_raw(f"    {'-'*45}\n")
            for i, t in enumerate(self.trades):
                pb_str = "Yes" if t.is_pullback else "No"
                aligned_str = "Yes" if t.trend_aligned else "No"
                self.diag.log_raw(f"    {i+1:<3} {t.trend_prob:<8.1%} {t.bounce_prob:<9.1%} {pb_str:<10} {aligned_str:<10}\n")
            
            self.diag.log_metric("backtest_total_trades", result.total_trades)
            self.diag.log_metric("backtest_win_rate", result.win_rate, warn_if='high')
            self.diag.log_metric("backtest_total_pnl", result.total_pnl)
            self.diag.log_metric("backtest_profit_factor", result.profit_factor)
            
            # Log position handling verification
            self.diag.log_raw(f"\n  ✓ VERIFIED: One position at a time (matches live trading)\n")
            
            # =====================================================
            # GRADE ANALYSIS - WHY ARE A-GRADES UNDERPERFORMING?
            # =====================================================
            self.diag.log_raw(f"\n\n{'='*80}\n")
            self.diag.log_raw(f"GRADE ANALYSIS - INVESTIGATING A-GRADE PERFORMANCE\n")
            self.diag.log_raw(f"{'='*80}\n")
            
            # Grade definitions reminder
            self.diag.log_raw(f"\n  Grade Definitions:\n")
            self.diag.log_raw(f"    A: bounce_prob > 0.6 AND trend_aligned AND is_pullback\n")
            self.diag.log_raw(f"    B: bounce_prob > 0.5 AND (trend_aligned OR is_pullback)\n")
            self.diag.log_raw(f"    C: Everything else\n")
            
            # Analyze each grade
            for grade in ['A', 'B', 'C']:
                signals = self._grade_analysis[grade]['signals']
                trades = self._grade_analysis[grade]['trades']
                
                self.diag.log_raw(f"\n  --- Grade {grade} Analysis ---\n")
                self.diag.log_raw(f"    Total signals: {len(signals)}\n")
                self.diag.log_raw(f"    Trades executed: {len(trades)}\n")
                
                if trades:
                    wins = [t for t in trades if t['win']]
                    losses = [t for t in trades if not t['win']]
                    win_rate = len(wins) / len(trades) * 100
                    
                    self.diag.log_raw(f"    Win rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)\n")
                    
                    # Analyze winning vs losing trades
                    if wins:
                        avg_trend_prob_wins = sum(t['trend_prob'] for t in wins) / len(wins)
                        avg_bounce_prob_wins = sum(t['bounce_prob'] for t in wins) / len(wins)
                        avg_dist_ema_wins = sum(t['dist_from_ema'] for t in wins) / len(wins)
                        pullback_pct_wins = sum(1 for t in wins if t['is_pullback']) / len(wins) * 100
                        aligned_pct_wins = sum(1 for t in wins if t['trend_aligned']) / len(wins) * 100
                        
                        self.diag.log_raw(f"\n    WINNING trades ({len(wins)}):\n")
                        self.diag.log_raw(f"      Avg trend_prob:   {avg_trend_prob_wins:.1%}\n")
                        self.diag.log_raw(f"      Avg bounce_prob:  {avg_bounce_prob_wins:.1%}\n")
                        self.diag.log_raw(f"      Avg dist_from_ema: {avg_dist_ema_wins:.2f} ATR\n")
                        self.diag.log_raw(f"      % in pullback:    {pullback_pct_wins:.0f}%\n")
                        self.diag.log_raw(f"      % trend aligned:  {aligned_pct_wins:.0f}%\n")
                        
                        # Exit reasons for wins
                        exit_reasons_wins = {}
                        for t in wins:
                            r = t['exit_reason']
                            exit_reasons_wins[r] = exit_reasons_wins.get(r, 0) + 1
                        self.diag.log_raw(f"      Exit reasons: {exit_reasons_wins}\n")
                    
                    if losses:
                        avg_trend_prob_losses = sum(t['trend_prob'] for t in losses) / len(losses)
                        avg_bounce_prob_losses = sum(t['bounce_prob'] for t in losses) / len(losses)
                        avg_dist_ema_losses = sum(t['dist_from_ema'] for t in losses) / len(losses)
                        pullback_pct_losses = sum(1 for t in losses if t['is_pullback']) / len(losses) * 100
                        aligned_pct_losses = sum(1 for t in losses if t['trend_aligned']) / len(losses) * 100
                        
                        self.diag.log_raw(f"\n    LOSING trades ({len(losses)}):\n")
                        self.diag.log_raw(f"      Avg trend_prob:   {avg_trend_prob_losses:.1%}\n")
                        self.diag.log_raw(f"      Avg bounce_prob:  {avg_bounce_prob_losses:.1%}\n")
                        self.diag.log_raw(f"      Avg dist_from_ema: {avg_dist_ema_losses:.2f} ATR\n")
                        self.diag.log_raw(f"      % in pullback:    {pullback_pct_losses:.0f}%\n")
                        self.diag.log_raw(f"      % trend aligned:  {aligned_pct_losses:.0f}%\n")
                        
                        # Exit reasons for losses
                        exit_reasons_losses = {}
                        for t in losses:
                            r = t['exit_reason']
                            exit_reasons_losses[r] = exit_reasons_losses.get(r, 0) + 1
                        self.diag.log_raw(f"      Exit reasons: {exit_reasons_losses}\n")
                    
                    # Direction analysis
                    longs = [t for t in trades if t['direction'] == 1]
                    shorts = [t for t in trades if t['direction'] == -1]
                    if longs:
                        long_wr = sum(1 for t in longs if t['win']) / len(longs) * 100
                        self.diag.log_raw(f"\n    LONG trades: {len(longs)}, win rate: {long_wr:.0f}%\n")
                    if shorts:
                        short_wr = sum(1 for t in shorts if t['win']) / len(shorts) * 100
                        self.diag.log_raw(f"    SHORT trades: {len(shorts)}, win rate: {short_wr:.0f}%\n")
                
                else:
                    self.diag.log_raw(f"    No trades executed for this grade\n")
            
            # Summary comparison
            self.diag.log_raw(f"\n  --- Summary Comparison ---\n")
            self.diag.log_raw(f"    {'Grade':<6} {'Signals':<10} {'Trades':<8} {'Win Rate':<10} {'Avg BounceP':<12}\n")
            self.diag.log_raw(f"    {'-'*50}\n")
            
            for grade in ['A', 'B', 'C']:
                signals = self._grade_analysis[grade]['signals']
                trades = self._grade_analysis[grade]['trades']
                
                if trades:
                    wr = sum(1 for t in trades if t['win']) / len(trades) * 100
                    avg_bp = sum(t['bounce_prob'] for t in trades) / len(trades) * 100
                else:
                    wr = 0
                    avg_bp = 0
                
                self.diag.log_raw(f"    {grade:<6} {len(signals):<10} {len(trades):<8} {wr:<10.1f}% {avg_bp:<12.1f}%\n")
            
            # Hypothesis about A-grade underperformance
            a_trades = self._grade_analysis['A']['trades']
            b_trades = self._grade_analysis['B']['trades']
            
            if a_trades and b_trades:
                a_avg_bounce = sum(t['bounce_prob'] for t in a_trades) / len(a_trades)
                b_avg_bounce = sum(t['bounce_prob'] for t in b_trades) / len(b_trades)
                
                self.diag.log_raw(f"\n  --- Hypothesis Analysis ---\n")
                self.diag.log_raw(f"    A-grade avg bounce_prob: {a_avg_bounce:.1%}\n")
                self.diag.log_raw(f"    B-grade avg bounce_prob: {b_avg_bounce:.1%}\n")
                
                if a_avg_bounce > b_avg_bounce:
                    self.diag.log_raw(f"\n    ⚠️ A-grades have HIGHER bounce_prob but LOWER win rate!\n")
                    self.diag.log_raw(f"    Possible causes:\n")
                    self.diag.log_raw(f"    1. Entry model (bounce_prob) is overfitted and unreliable\n")
                    self.diag.log_raw(f"    2. 'is_pullback' requirement catches reversals, not pullbacks\n")
                    self.diag.log_raw(f"    3. High-confidence signals occur at market turning points\n")
                    self.diag.log_raw(f"    4. Small sample size - need more trades to validate\n")
        
        return result


def print_backtest_results(result: BacktestResult):
    """Pretty print backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"\n  Total Trades:     {result.total_trades}")
    print(f"  Winning Trades:   {result.winning_trades}")
    print(f"  Losing Trades:    {result.losing_trades}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    
    print(f"\n  Total P&L:        ${result.total_pnl:,.2f}")
    print(f"  Total Return:     {result.total_pnl_percent:.2f}%")
    print(f"  Avg Win:          ${result.avg_win:,.2f}")
    print(f"  Avg Loss:         ${result.avg_loss:,.2f}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    
    print(f"\n  Max Drawdown:     ${result.max_drawdown:,.2f}")
    print(f"  Max DD %:         {result.max_drawdown_percent:.2f}%")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    
    print("\n  Performance by Signal Grade:")
    for grade in ['A', 'B', 'C']:
        count = result.trades_by_grade.get(grade, 0)
        wr = result.win_rate_by_grade.get(grade, 0)
        print(f"    Grade {grade}: {count} trades, {wr:.1%} win rate")
    
    print("=" * 60)


def run_backtest(data, models, feature_cols, config=DEFAULT_CONFIG, **kwargs) -> BacktestResult:
    """Convenience function to run a backtest."""
    backtester = SimpleBacktester(models, config, **kwargs)
    result = backtester.run(data, feature_cols)
    print_backtest_results(result)
    return result


if __name__ == "__main__":
    print("Backtest module loaded successfully")
