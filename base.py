"""
Base classes and performance metrics for CTA strategies.
"""

from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


###### Performance Metrics

class PerformanceMetrics:
    
    """Computes and stores standard performance metrics from a returns series."""

    def __init__(
        self,
        total_return: float = 0.0,
        annualized_return: float = 0.0,
        annualized_volatility: float = 0.0,
        sharpe_ratio: float = 0.0,
        max_drawdown: float = 0.0,
        calmar_ratio: float = 0.0,
        hit_rate: float = 0.0,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        profit_factor: float = 0.0):

        self.total_return = total_return
        self.annualized_return = annualized_return
        self.annualized_volatility = annualized_volatility
        self.sharpe_ratio = sharpe_ratio
        self.max_drawdown = max_drawdown
        self.calmar_ratio = calmar_ratio
        self.hit_rate = hit_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.profit_factor = profit_factor

    @classmethod
    def from_returns(cls, returns: pd.DataFrame, returns_col: str = 'return', 
                     risk_free_rate: float = 0.0, periods_per_year: int = 252) -> "PerformanceMetrics":
        """
        Compute metrics from a simple-returns series.

        Parameters
        ----------
        cls : type
            Class of the performance metrics.
        returns : pd.DataFrame
            Daily (or periodic) simple returns.
        returns_col : str
            Column name of the returns series.
        risk_free_rate : float
            Annualized risk-free rate for Sharpe computation.
        periods_per_year : int
            Trading periods per year (252 for daily, 12 for monthly).
        """
        returns = returns.dropna()
        if len(returns) < 2:
            return cls()

        # Equity curve
        equity = (1 + returns[returns_col]).cumprod()

        # Total & annualized return
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = len(returns) / periods_per_year
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Annualized volatility
        ann_vol = returns[returns_col].std() * np.sqrt(periods_per_year)

        # Sharpe ratio
        excess = returns[returns_col].mean() - risk_free_rate / periods_per_year
        sharpe = (
            excess / returns[returns_col].std() * np.sqrt(periods_per_year)
            if returns[returns_col].std() > 0
            else 0.0
        )

        # Max drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

        # Hit rate, average win/loss, profit factor
        wins = returns[returns[returns_col] > 0][returns_col]
        losses = returns[returns[returns_col] < 0][returns_col]
        hit_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        profit_factor = (
            wins.sum() / abs(losses.sum())
            if len(losses) > 0 and losses.sum() != 0
            else float("inf")
        )

        return cls(
            total_return=total_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            hit_rate=hit_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
        )

    def summary(self) -> pd.Series:
        """Return a readable summary as a pandas Series."""
        return pd.Series(
            {
                "Total Return": f"{self.total_return:.2%}",
                "Ann. Return": f"{self.annualized_return:.2%}",
                "Ann. Volatility": f"{self.annualized_volatility:.2%}",
                "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
                "Max Drawdown": f"{self.max_drawdown:.2%}",
                "Calmar Ratio": f"{self.calmar_ratio:.3f}",
                "Hit Rate": f"{self.hit_rate:.2%}",
                "Avg Win": f"{self.avg_win:.4%}",
                "Avg Loss": f"{self.avg_loss:.4%}",
                "Profit Factor": f"{self.profit_factor:.3f}",
            }
        )


###### Abstract Strategy

class Strategy(ABC):
    """
    Abstract base class for all CTA-style trading strategies.
    Ensures that all strategies have the same interface.

    Subclasses must implement:
        - generate_signals: produce raw directional signals
        - get_positions:    produce final position sizes (after risk scaling)
    """

    def __init__(self, name: str = "Strategy"):
        self.name = name

    @abstractmethod
    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        """
        Generate raw trading signals from price data.

        Returns
        -------
        pd.Series
            Signal values (positive = long, negative = short).
        """
        ...

    @abstractmethod
    def get_positions(self, prices: pd.Series, **kwargs) -> pd.Series:
        """
        Generate final position sizes (incorporating risk management).

        Returns
        -------
        pd.Series
            Position sizes (e.g., +1 = fully long, -1 = fully short).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
