"""
Classical CTA-Style Futures and FX Strategies

A modularized Python package implementing the following strategies:

- **Long-Only Benchmark**
- **Volatility Targeting**
- **Time-Series Momentum (TSMOM)**
- **SMA Crossover**
- **MACD** (single & combined multi-scale)
- **FX Carry Trading**

Usage
-----
>>> from cta_strategies import TSMOMStrategy, BacktestEngine
>>> strategy = TSMOMStrategy(lookback_k=252, vol_target=0.15)
>>> engine = BacktestEngine(transaction_cost_bps=5)
>>> result = engine.run(strategy, prices)
>>> print(result.summary())
>>> result.plot()
"""

# Strategies
from .strategies import (
    LongOnlyBenchmark,
    TSMOMStrategy,
    SMACrossoverStrategy,
    MACDStrategy,
    CarryStrategy,
)

# Base
from .base import Strategy, PerformanceMetrics

# Data utilities
from .data import (
    build_nearest_futures,
    build_continuous_futures,
    compute_returns,
)

# Signal generators
from .signals import (
    time_series_momentum_signal,
    sma_crossover_signal,
    macd_signal,
    combined_macd_signal,
    carry_signal,
)

# Risk management
from .risk import (
    realized_volatility,
    volatility_target_sizing,
    leverage_cap,
    drawdown_stop,
)

# Backtester
from .backtester import BacktestEngine, BacktestResult

__all__ = [
    # Strategies
    "LongOnlyBenchmark",
    "TSMOMStrategy",
    "SMACrossoverStrategy",
    "MACDStrategy",
    "CarryStrategy",
    # Base
    "Strategy",
    "PerformanceMetrics",
    # Data
    "build_nearest_futures",
    "build_continuous_futures",
    "compute_returns",
    # Signals
    "time_series_momentum_signal",
    "sma_crossover_signal",
    "macd_signal",
    "combined_macd_signal",
    "carry_signal",
    # Risk
    "realized_volatility",
    "volatility_target_sizing",
    "leverage_cap",
    "drawdown_stop",
    # Backtester
    "BacktestEngine",
    "BacktestResult",
]

__version__ = "0.1.0"
