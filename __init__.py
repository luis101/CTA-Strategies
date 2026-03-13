"""
Classical CTA-Style Futures and FX Strategies
==============================================

A modularized Python package implementing the strategies described in
Section 5.1.1 of *Deep Learning in Quantitative Trading*:

- **Long-Only Benchmark**
- **Volatility Targeting** (Eq. 70)
- **Time-Series Momentum (TSMOM)** (Eq. 71)
- **SMA Crossover** (Eq. 72)
- **MACD** (single & combined multi-scale) (Eqs. 73–74)
- **FX Carry Trading** (Eqs. 75–76)

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
    tsmom_signal,
    sma_crossover_signal,
    macd_signal,
    combined_macd_signal,
    carry_signal,
)

# Risk management
from .risk import (
    estimate_volatility,
    volatility_target_sizing,
    apply_leverage_cap,
    apply_drawdown_stop,
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
    "tsmom_signal",
    "sma_crossover_signal",
    "macd_signal",
    "combined_macd_signal",
    "carry_signal",
    # Risk
    "estimate_volatility",
    "volatility_target_sizing",
    "apply_leverage_cap",
    "apply_drawdown_stop",
    # Backtester
    "BacktestEngine",
    "BacktestResult",
]

__version__ = "0.1.0"
