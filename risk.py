"""
Risk management tools for CTA strategies.

Approaches:
    - Volatility targeting (position sizing)
    - Leverage caps
    - Drawdown stop-loss
"""

import numpy as np
import pandas as pd


def realized_volatility(returns: pd.Series, lookback: int = 60, method: str = "ewm", 
                        annualization_factor: int = 252) -> pd.Series:
    """
    Estimate rolling annualized volatility

    Parameters
    ----------
    returns : pd.Series
        Simple returns
    lookback : int
        Window for rolling std or EWMA span
    method : str
        'rolling' for simple rolling std, 'ewm' for exponentially weighted
    annualization_factor : int
        Number of periods per year

    Returns
    -------
    pd.Series
        Annualized volatility estimate
    """
    if method == "ewm":
        vol = returns.ewm(span=lookback).std() * np.sqrt(annualization_factor)
    else:
        vol = returns.rolling(window=lookback).std() * np.sqrt(annualization_factor)
    return vol


def volatility_target_sizing(signals: pd.Series, returns: pd.Series, target_vol: float = 0.15,
                            vol_lookback: int = 60, vol_method: str = "ewm",
                            annualization_factor: int = 252) -> pd.Series:
    """
    Scale positions by volatility targeting

    A = sigma_target / sigma_current

    Positions = signal × A

    Parameters
    ----------
    signals : pd.Series
        Raw directional signals (±1 or continuous)
    returns : pd.Series
        Historical returns for volatility estimation
    target_vol : float
        Annualized target volatility (e.g., 0.15 for 15%)
    vol_lookback : int
        Look-back for volatility estimation
    vol_method : str
        'rolling' or 'ewm'
    annualization_factor : int
        Periods per year

    Returns
    -------
    pd.Series
        Volatility-scaled position sizes.
    """
    current_vol = realized_volatility(
        returns, lookback=vol_lookback, method=vol_method,
        annualization_factor=annualization_factor,
    )
    # Avoid division by zero / near-zero vol
    current_vol = current_vol.replace(0, np.nan).ffill()
    current_vol = current_vol.clip(lower=1e-6)

    adjustment = target_vol / current_vol
    return signals * adjustment


def leverage_cap(positions: pd.Series, max_leverage: float = 2.0) -> pd.Series:
    """
    Clip absolute position sizes to a maximum leverage

    Parameters
    ----------
    positions : pd.Series
        Position sizes (can exceed 1 in absolute value)
    max_leverage : float
        Maximum allowed absolute position size

    Returns
    -------
    pd.Series
        Capped positions
    """
    return positions.clip(lower=-max_leverage, upper=max_leverage)


def drawdown_stop(equity: pd.Series, positions: pd.Series, max_drawdown_pct: float = 0.20) -> pd.Series:
    """
    Set positions to zero when the drawdown from peak exceeds a threshold

    Parameters
    ----------
    equity : pd.Series
        Running equity curve
    positions : pd.Series
        Current positions
    max_drawdown_pct : float
        Maximum drawdown threshold (e.g., 0.20 for 20%)

    Returns
    -------
    pd.Series
        Positions with drawdown stop applied
    """
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    # Where drawdown exceeds threshold, flatten positions
    stopped = positions.copy()
    stopped[drawdown < -max_drawdown_pct] = 0.0
    return stopped
