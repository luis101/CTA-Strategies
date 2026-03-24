"""
Signals for CTA-style trading strategies

Implements all signal formulas from:
    - Time-series momentum (TSMOM)
    - Simple moving average crossover (SMA)
    - MACD (single and combined)
    - Carry signal
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


###### Time-Series Momentum

def time_series_momentum_signal(returns: pd.Series, lookback_k: int = 252) -> pd.Series:
    """
    Time-series momentum signal: sign of the cumulative return
    over the past lookback_k periods.

    Y_t = sign(r_{t-k:t})

    Parameters
    ----------
    returns : pd.Series
        Simple returns series.
    lookback_k : int
        Number of periods to look back (default: 252 ≈ 12 months daily).

    Returns
    -------
    pd.Series
        Signal: +1 (long), -1 (short), or 0 (flat).
    """
    cum_ret = returns.rolling(window=lookback_k).sum()
    return np.sign(cum_ret)


###### Simple Moving Average Crossover

def sma_crossover_signal(prices: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    """
    Simple moving average crossover signal.

    Y_t = SMA(t, K_1) - SMA(t, K_2)

    Positive → long, negative → short.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    short_window : int
        Short SMA look-back period K1.
    long_window : int
        Long SMA look-back period K2 (K2 > K1).

    Returns
    -------
    pd.Series
        Signal values (raw difference; sign determines direction).
    """
    sma_short = prices.rolling(window=short_window).mean()
    sma_long = prices.rolling(window=long_window).mean()
    return sma_short - sma_long


###### MACD

def macd_signal(prices: pd.Series, short_span: int = 12, long_span: int = 26) -> pd.Series:
    """
    MACD signal (single pair of time-scales).

    MACD(t, S, L) = EWMA(t, S) - EWMA(t, L)

    where α = 2 / (span + 1).

    Parameters
    ----------
    prices : pd.Series
        Price series.
    short_span : int
        Short EWMA span *S*.
    long_span : int
        Long EWMA span *L* (L > S).

    Returns
    -------
    pd.Series
        MACD values.
    """
    ewma_short = prices.ewm(span=short_span, adjust=False).mean()
    ewma_long = prices.ewm(span=long_span, adjust=False).mean()
    return ewma_short - ewma_long


def combined_macd_signal(prices: pd.Series, pairs: Optional[List[Tuple[int, int]]] = None) -> pd.Series:
    """
    Combined MACD signal: sum of multiple MACD signals at different time-scales.

    Y_t = sum_{k=1}^{K} Y_t(S_k, L_k)

    Default pairs: S ∈ {8, 16, 32}, L ∈ {24, 48, 96}.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    pairs : list of (short_span, long_span) tuples, optional
        MACD time-scale pairs.  Defaults to [(8,24), (16,48), (32,96)].

    Returns
    -------
    pd.Series
        Combined MACD signal.
    """
    if pairs is None:
        pairs = [(8, 24), (16, 48), (32, 96)]

    combined = pd.Series(0.0, index=prices.index)
    for s, l in pairs:
        combined += macd_signal(prices, short_span=s, long_span=l)
    return combined


###### Carry

def carry_signal(rate_a: pd.Series, rate_b: pd.Series) -> pd.Series:
    """
    Carry (interest rate differential) signal for FX pairs.

    IRD = i_A - i_B

    Positive IRD → long Currency A / short Currency B.

    Parameters
    ----------
    rate_a : pd.Series
        Interest rate series for the high-yield currency.
    rate_b : pd.Series
        Interest rate series for the funding (low-yield) currency.

    Returns
    -------
    pd.Series
        Signal: +1 (go long A / short B) or -1 (reverse).
    """
    ird = rate_a - rate_b
    return np.sign(ird)
