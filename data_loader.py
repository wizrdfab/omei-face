"""
Data loader for raw trade data from Bybit.
Handles loading multiple CSV files and basic preprocessing.
"""
# --- 1. PERFORMANCE SETTINGS (Must be before imports) ---
import os
# Force internal libraries to use 1 thread. 
# We handle parallelism via Multiprocessing, not internal threading.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import concurrent.futures
import multiprocessing
import gc
import sys
from config import DataConfig


# --- GLOBAL SHARED MEMORY STORAGE ---
_shared_trades_df = None


def _process_file(file_path: Path, use_cols: list, dtypes: dict, config: DataConfig, sample_rate: float) -> pd.DataFrame:
    """
    Helper function to process a single file in a separate process.
    """
    try:
        # Load only needed columns with optimized dtypes
        df = pd.read_csv(
            file_path, 
            usecols=lambda c: c in use_cols or c == 'symbol',
            dtype=dtypes
        )
        
        # Sample if needed
        if sample_rate < 1.0:
            df = df.sample(frac=sample_rate, random_state=42)
        
        # Sort this file's trades (Mergesort is stable)
        df = df.sort_values(config.timestamp_col, kind='mergesort')
        
        return df
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return pd.DataFrame()


def load_trades(config: DataConfig, verbose: bool = True, sample_rate: float = 1.0) -> pd.DataFrame:
    """
    Load all trade CSV files from the data directory concurrently.
    Guarantees DETERMINISTIC order using stable sorts and ordered processing.
    """
    data_path = Path(config.data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find all matching files (Sorted ensures deterministic input order)
    files = sorted(data_path.glob(config.file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{config.file_pattern}' in {data_path}")
    
    if verbose:
        print(f"Found {len(files)} trade files:")
        for f in files:
            print(f"  - {f.name}")
    
    use_cols = [
        config.timestamp_col,
        config.price_col,
        config.size_col,
        config.side_col,
        config.tick_direction_col,
    ]

    dtypes = {
        config.price_col: 'float32',
        config.size_col: 'float32',
    }
    
    all_trades = []
    total_original = 0
    
    if verbose:
        print(f"  Starting concurrent load on {len(files)} files...")

    # Use ProcessPoolExecutor for CPU-bound CSV parsing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks in order
        future_to_index = {
            executor.submit(_process_file, f, use_cols, dtypes, config, sample_rate): i 
            for i, f in enumerate(files)
        }
        
        results = [None] * len(files)
        
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            file_name = files[index].name
            try:
                df = future.result()
                if not df.empty:
                    total_original += (len(df) / sample_rate) if sample_rate < 1.0 else len(df)
                    results[index] = df 
                    if verbose:
                        print(f"  Loaded {file_name}: {len(df):,} trades")
                else:
                    results[index] = pd.DataFrame()
                
            except Exception as exc:
                print(f"  {file_name} generated an exception: {exc}")
                results[index] = pd.DataFrame()

    all_trades = [df for df in results if df is not None and not df.empty]
    gc.collect()
    
    if verbose:
        print("  Merging files...")
    
    if not all_trades:
        raise ValueError("No trades were loaded from the files.")

    trades = pd.concat(all_trades, ignore_index=True)
    del all_trades
    
    # Final sort: ESSENTIAL to use 'mergesort' (stable)
    trades = trades.sort_values(config.timestamp_col, kind='mergesort').reset_index(drop=True)
    
    if verbose:
        print(f"\nTotal trades: {len(trades):,}" + 
              (f" (sampled from approx {int(total_original):,})" if sample_rate < 1.0 else ""))
        print(f"Date range: {pd.to_datetime(trades[config.timestamp_col].min(), unit='s')} "
              f"to {pd.to_datetime(trades[config.timestamp_col].max(), unit='s')}")
        
        mem_mb = trades.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Memory usage: {mem_mb:.1f} MB")
    
    return trades


def preprocess_trades(trades: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """
    Preprocess raw trade data.
    """
    df = trades.copy()
    
    df['datetime'] = pd.to_datetime(df[config.timestamp_col], unit='s')
    df['side_num'] = df[config.side_col].map({'Buy': 1, 'Sell': -1})
    
    tick_map = {
        'PlusTick': 1, 'ZeroPlusTick': 0.5, 'MinusTick': -1, 'ZeroMinusTick': -0.5
    }
    df['tick_dir_num'] = df[config.tick_direction_col].map(tick_map).fillna(0)
    
    df['value'] = df[config.price_col] * df[config.size_col]
    df['signed_size'] = df[config.size_col] * df['side_num']
    df['signed_value'] = df['value'] * df['side_num']
    
    return df


def aggregate_to_bars(
    trades: pd.DataFrame, 
    timeframe_seconds: int,
    config: DataConfig
) -> pd.DataFrame:
    """
    Aggregate raw trades into OHLCV bars.
    MEMORY OPTIMIZED: Avoids df.copy() to save RAM during multiprocessing.
    """
    # Calculate grouping key
    bar_time_series = (trades[config.timestamp_col] // timeframe_seconds) * timeframe_seconds
    
    # Aggregate directly on the original trades df
    bars = trades.groupby(bar_time_series).agg({
        config.price_col: ['first', 'max', 'min', 'last'],
        config.size_col: 'sum',
        'value': 'sum',
        'side_num': 'sum', 
        'signed_size': 'sum', 
        'signed_value': 'sum', 
        'tick_dir_num': 'mean', 
        config.timestamp_col: 'count', 
    })
    
    # Flatten column names
    bars.columns = [
        'open', 'high', 'low', 'close',
        'volume', 'value',
        'net_side', 'net_volume', 'net_value',
        'avg_tick_dir', 'trade_count'
    ]
    
    bars.index.name = 'bar_time'
    bars = bars.reset_index()
    bars['datetime'] = pd.to_datetime(bars['bar_time'], unit='s')
    
    # Calculate additional metrics
    bars['buy_volume'] = (bars['volume'] + bars['net_volume']) / 2
    bars['sell_volume'] = (bars['volume'] - bars['net_volume']) / 2
    bars['buy_sell_imbalance'] = bars['net_volume'] / bars['volume'].replace(0, np.nan)
    
    bars['vwap'] = bars['value'] / bars['volume'].replace(0, np.nan)
    bars['trade_intensity'] = bars['trade_count'] / timeframe_seconds
    bars['avg_trade_size'] = bars['volume'] / bars['trade_count'].replace(0, np.nan)
    
    bars = bars.fillna(method='ffill')
    
    return bars


def _generate_bars_worker(seconds: int, name: str, config: DataConfig):
    """
    Worker function. 
    Reads from the GLOBAL variable `_shared_trades_df` to avoid pickling overhead.
    """
    global _shared_trades_df
    if _shared_trades_df is None:
        raise ValueError(f"Worker {name} found empty shared dataframe!")
    
    # LOGGING: Print here so user sees tasks starting concurrently
    # flush=True is required to force the output to appear immediately in the terminal
    print(f"  [Worker] > Starting {name} bar creation...", flush=True)
    
    bars = aggregate_to_bars(_shared_trades_df, seconds, config)
    
    print(f"  [Worker] V Finished {name} ({len(bars):,} bars)", flush=True)
    return name, bars


def create_multi_timeframe_bars(
    trades: pd.DataFrame,
    timeframes_seconds: List[int],
    timeframe_names: List[str],
    config: DataConfig
) -> dict:
    """
    Create bars for multiple timeframes concurrently using Multiprocessing + Shared Memory.
    """
    global _shared_trades_df
    bars_dict = {}
    
    # 1. Store trades in global variable for Zero-Copy access by workers
    _shared_trades_df = trades
    
    # Detect optimal CPU count (leaving 1 core free for OS/SSH if possible)
    cpu_count = multiprocessing.cpu_count()
    # If we have 4 cores, use 3. If 2, use 2. If 1, use 1.
    MAX_WORKERS = max(1, cpu_count - 1) if cpu_count > 2 else cpu_count
    
    print(f"Starting TRUE concurrent bar creation for {len(timeframe_names)} timeframes...")
    print(f"  (Using Fork-based Multiprocessing with {MAX_WORKERS} workers)")
    print(f"  (OMP/MKL threading disabled per process to avoid CPU contention)")

    try:
        # 2. Use 'fork' context. 
        # This is CRITICAL for Linux/Mac to share memory without copying.
        ctx = multiprocessing.get_context('fork')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as executor:
            # Note: We do NOT pass 'trades' as an argument here.
            future_to_name = {
                executor.submit(_generate_bars_worker, sec, name, config): name
                for sec, name in zip(timeframes_seconds, timeframe_names)
            }
            
            # We wait for completion here, but the print statements inside 
            # _generate_bars_worker will show you the concurrency in real-time.
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    tf_name, bars = future.result()
                    bars_dict[tf_name] = bars
                    # Removed the print here to avoid confusion. 
                    # The worker handles printing now.
                except Exception as exc:
                    print(f"  -> {name} generated an exception: {exc}")
                    
    finally:
        # 3. Cleanup global memory
        _shared_trades_df = None
    
    return bars_dict


if __name__ == "__main__":
    from config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.data
    config.data_dir = Path("./data")
    
    print("Loading trades (Deterministic)...")
    trades = load_trades(config)
    
    print("\nPreprocessing...")
    trades = preprocess_trades(trades, config)
    
    print("\nCreating multi-timeframe bars...")
    bars = create_multi_timeframe_bars(
        trades,
        DEFAULT_CONFIG.features.timeframes,
        DEFAULT_CONFIG.features.timeframe_names,
        config
    )
    
    print("\nSample 5m bars:")
    if '5m' in bars:
        print(bars['5m'].head(10))
    else:
        first_key = list(bars.keys())[0]
        print(f"(5m not found, showing {first_key})")
        print(bars[first_key].head(10))
