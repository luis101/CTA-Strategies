"""
End-to-end demo of the CTA strategies package

This example script generates synthetic futures prices (geometric Brownian motion with embedded
trend), runs every strategy through the backtester, and prints a comparison table.
"""

import sys
import os
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# Ensure the parent directory is on the path so we can import cta_strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cta_strategies import (
    LongOnlyBenchmark,
    TSMOMStrategy,
    SMACrossoverStrategy,
    MACDStrategy,
    CarryStrategy,
    BacktestEngine,
)
from cta_strategies.data import build_continuous_futures, build_nearest_futures


###### Synthetic Data

def generate_synthetic_prices(
    n_days: int = 2620,  # ~10 years
    mu: float = 0.05,    # annual drift
    sigma: float = 0.20, # annual volatility
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a DataFrame of hypothetical futures prices
    
    Creates a simulated price series using geometric Brownian motion with an embedded trend, 
    and structures it into 24 futures contracts expiring on the last Friday of each month up to 2 years out.

    We can just test using these 24 contracts, as they are continuously rolled.
    """
    np.random.seed(seed)
    dt = 1 / 252
    
    # Base dates (2016-03-02 to 2026-01-02) 10 years of data
    dates = pd.bdate_range(start="2016-03-02", periods=n_days)
    
    mid = n_days // 2
    drifts = np.concatenate([
        np.full(mid, mu * 1.5),     # strong uptrend
        np.full(n_days - mid, mu * 0.3),  # weaker trend
    ])
    
    # Linearized geometric Brownian motion with drift and volatility
    log_returns = (drifts * dt
                   - 0.5 * sigma**2 * dt
                   + sigma * np.sqrt(dt) * np.random.randn(n_days))
    spot_prices = 100 * np.exp(np.cumsum(log_returns))
    
    # Expiry dates (last Friday of the month) up to 3 years after the max date
    max_date = dates.max() + pd.DateOffset(years=3)
    min_date = dates.min() - pd.DateOffset(months=1)
    
    all_days = pd.date_range(min_date, max_date, freq='D')
    fridays = all_days[all_days.weekday == 4]
    last_fridays = pd.Series(fridays).groupby(fridays.to_period('M')).max().values
    expiry_dates = pd.DatetimeIndex(last_fridays)
    
    rows = []
    expiry_array = expiry_dates.values
    
    for date, spot in zip(dates, spot_prices):
        # Find the index of the first expiry date >= current date
        idx = np.searchsorted(expiry_array, date.to_datetime64())
        
        # Take the next 24 expirations
        active_expiries = expiry_array[idx:idx+24]
        
        for contract_idx, exp in enumerate(active_expiries):
            # Simplistic pricing model for out-of-the-money contracts
            future_price = spot * (1 + 0.001 * contract_idx)
            month_str = pd.Timestamp(exp).strftime('%Y%m')
            
            rows.append({
                'date': date,
                'expire_date': exp,
                'future_contract': f"FUT_{month_str}",
                'price': future_price
            })
            
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['expire_date'] = pd.to_datetime(df['expire_date'])
    return df


def generate_synthetic_rates(
    index: pd.DatetimeIndex,
    base_rate_a: float = 0.05,
    base_rate_b: float = 0.01,
    seed: int = 123,
) -> tuple:
    """Generate synthetic interest rate series for two currencies."""
    np.random.seed(seed)
    n = len(index)

    rate_a = base_rate_a + 0.005 * np.cumsum(np.random.randn(n) * 0.01)
    rate_b = base_rate_b + 0.003 * np.cumsum(np.random.randn(n) * 0.01)

    return (
        pd.Series(rate_a, index=index, name="Rate A"),
        pd.Series(rate_b, index=index, name="Rate B"),
    )


###### Main function

def main():
    print("=" * 70)
    print("  CTA-Style Futures & FX Strategies — Demo")
    print("=" * 70)
    print()

    # Generate data
    raw_futures = generate_synthetic_prices()
    price_data = build_continuous_futures(raw_futures, method="ratio")
    prices = price_data['price']
    
    rate_a, rate_b = generate_synthetic_rates(prices.index)

    print(f"Synthetic raw futures contracts: {len(raw_futures)} rows")
    print(f"Continuous series length: {len(prices)} days "
          f"({prices.index[0].date()} — {prices.index[-1].date()})")

    # Define strategies
    strategies = [
        LongOnlyBenchmark(name="Long-Only (no volatility target)"),
        LongOnlyBenchmark(vol_target=0.15, name="Long-Only (volatility target=15%)"),
        TSMOMStrategy(lookback_k=252, vol_target=0.15, name="Time Series Momentum 12m"),
        TSMOMStrategy(lookback_k=63, vol_target=0.15, name="Time Series Momentum 3m"),
        SMACrossoverStrategy(
            short_window=50, long_window=200, vol_target=0.15,
            name="SMA Crossover (50/200)",
        ),
        MACDStrategy(
            use_combined=True, vol_target=0.15, name="MACD Combined",
        ),
        MACDStrategy(
            use_combined=False, short_span=12, long_span=26,
            vol_target=0.15, name="MACD (12/26)",
        ),
        CarryStrategy(vol_target=0.15, name="FX Carry"),
    ]

    # Prepare kwargs (carry strategy needs rate series)
    kwargs_list = [{}] * 7 + [{"rate_a": rate_a, "rate_b": rate_b}]

    # Run backtests
    backtest_engine = BacktestEngine(transaction_cost_bps=5)
    comparison = backtest_engine.compare(strategies, prices, kwargs_list=kwargs_list)

    print("-- Performance Comparison " + "-" * 44)
    print()
    print(comparison.to_string())
    print()
    print("=" * 70)

    # Plot individual results
    matplotlib.use("Agg")  # non-interactive backend

    fig, ax = plt.subplots(figsize=(14, 6))
    for strategy, kw in zip(strategies, kwargs_list):
        result = backtest_engine.run(strategy, prices, **kw)
        result.equity_value.plot(ax=ax, label=strategy.name, linewidth=1.0)

    ax.set_title("Equity Values — All Strategies", fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (starting value = 1.0)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "equity_values.png")
    fig.savefig(out_path, dpi=150)
    print(f"Equity values plot saved to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
