"""
Backtesting engine for CTA strategies.

Runs strategies against historical price data and produces performance
reports with equity curves, drawdowns, and key metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .base import Strategy, PerformanceMetrics
from .data import compute_returns
from .risk import apply_drawdown_stop


@dataclass
class BacktestResult:
    """Container for backtesting results."""

    strategy_name: str
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    signals: pd.Series
    metrics: PerformanceMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.Series:
        """Return a readable summary of performance metrics."""
        return self.metrics.summary()

    def plot(self, figsize: tuple = (14, 8), show: bool = True):
        """
        Plot equity curve and drawdown.

        Parameters
        ----------
        figsize : tuple
            Figure size.
        show : bool
            If True, call plt.show().
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.suptitle(f"Backtest: {self.strategy_name}", fontsize=14,
                     fontweight="bold")

        # Equity curve
        ax1 = axes[0]
        self.equity_curve.plot(ax=ax1, color="#2196F3", linewidth=1.2)
        ax1.set_ylabel("Equity")
        ax1.set_title("Equity Curve")
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(
            self.equity_curve.index,
            1.0,
            self.equity_curve.values,
            alpha=0.08,
            color="#2196F3",
        )

        # Drawdown
        ax2 = axes[1]
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        drawdown.plot(ax=ax2, color="#F44336", linewidth=0.8)
        ax2.fill_between(drawdown.index, 0, drawdown.values,
                         alpha=0.2, color="#F44336")
        ax2.set_ylabel("Drawdown")
        ax2.set_title("Drawdown")
        ax2.grid(True, alpha=0.3)

        # Positions
        ax3 = axes[2]
        self.positions.plot(ax=ax3, color="#4CAF50", linewidth=0.8)
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax3.set_ylabel("Position")
        ax3.set_title("Position Size")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return fig


class BacktestEngine:
    """
    Vectorized backtesting engine for CTA strategies.

    Parameters
    ----------
    transaction_cost_bps : float
        One-way transaction cost in basis points (default: 5 bps).
    max_drawdown_stop : float, optional
        If set, flatten positions when drawdown exceeds this threshold.
    risk_free_rate : float
        Annualized risk-free rate for Sharpe calculation.
    periods_per_year : int
        Trading periods per year (252 for daily).
    """

    def __init__(
        self,
        transaction_cost_bps: float = 5.0,
        max_drawdown_stop: Optional[float] = None,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ):
        self.tc_bps = transaction_cost_bps
        self.max_drawdown_stop = max_drawdown_stop
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def run(
        self,
        strategy: Strategy,
        prices: pd.Series,
        **kwargs,
    ) -> BacktestResult:
        """
        Run a backtest for a given strategy and price series.

        Parameters
        ----------
        strategy : Strategy
            A strategy instance implementing :meth:`get_positions`.
        prices : pd.Series
            Price series to backtest against.
        **kwargs
            Additional keyword arguments passed to
            ``strategy.get_positions()``.

        Returns
        -------
        BacktestResult
            Full backtest results with equity curve, positions, and metrics.
        """
        # Generate positions (signal + risk sizing)
        positions = strategy.get_positions(prices, **kwargs)

        # Compute asset returns
        asset_returns = compute_returns(prices)

        # Align positions and returns
        positions = positions.reindex(asset_returns.index).fillna(0)

        # Lag positions by 1 (trade on signal, realize return next period)
        lagged_positions = positions.shift(1).fillna(0)

        # Strategy gross returns
        strategy_returns = lagged_positions * asset_returns

        # Transaction costs (proportional to position change)
        position_changes = lagged_positions.diff().abs()
        tc = position_changes * (self.tc_bps / 10_000)
        strategy_returns = strategy_returns - tc

        # Build equity curve
        equity = (1 + strategy_returns.fillna(0)).cumprod()

        # Optional drawdown stop
        if self.max_drawdown_stop is not None:
            lagged_positions = apply_drawdown_stop(
                equity, lagged_positions, max_drawdown_pct=self.max_drawdown_stop,
            )
            # Recompute with stopped positions
            strategy_returns = lagged_positions * asset_returns
            position_changes = lagged_positions.diff().abs()
            tc = position_changes * (self.tc_bps / 10_000)
            strategy_returns = strategy_returns - tc
            equity = (1 + strategy_returns.fillna(0)).cumprod()

        # Compute metrics
        clean_returns = strategy_returns.dropna()
        metrics = PerformanceMetrics.from_returns(
            clean_returns,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
        )

        # Generate raw signals for reference
        try:
            signals = strategy.generate_signals(prices, **kwargs)
        except Exception:
            signals = pd.Series(np.nan, index=prices.index)

        return BacktestResult(
            strategy_name=strategy.name,
            equity_curve=equity,
            returns=strategy_returns,
            positions=lagged_positions,
            signals=signals,
            metrics=metrics,
        )

    def compare(
        self,
        strategies: list,
        prices: pd.Series,
        kwargs_list: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Run multiple strategies and compare their performance.

        Parameters
        ----------
        strategies : list[Strategy]
            List of strategy instances.
        prices : pd.Series
            Shared price series.
        kwargs_list : list[dict], optional
            Per-strategy keyword arguments.

        Returns
        -------
        pd.DataFrame
            Comparison table with one row per strategy.
        """
        if kwargs_list is None:
            kwargs_list = [{}] * len(strategies)

        summaries = {}
        for strat, kw in zip(strategies, kwargs_list):
            result = self.run(strat, prices, **kw)
            summaries[strat.name] = result.summary()

        return pd.DataFrame(summaries)
