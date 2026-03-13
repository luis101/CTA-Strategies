"""
Concrete CTA strategy implementations.

Each strategy inherits from :class:`base.Strategy` and combines a signal
generator with optional volatility targeting.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from .base import Strategy
from .signals import (
    tsmom_signal,
    sma_crossover_signal,
    macd_signal,
    combined_macd_signal,
    carry_signal,
)
from .risk import volatility_target_sizing, apply_leverage_cap
from .data import compute_returns


class LongOnlyBenchmark(Strategy):
    """
    Long-only benchmark strategy.

    Always holds a full long position (+1).  Serves as the baseline for
    evaluating active strategies.  Optionally applies volatility targeting.
    """

    def __init__(
        self,
        vol_target: Optional[float] = None,
        vol_lookback: int = 60,
        name: str = "Long-Only Benchmark",
    ):
        super().__init__(name=name)
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        return pd.Series(1.0, index=prices.index)

    def get_positions(self, prices: pd.Series, **kwargs) -> pd.Series:
        signals = self.generate_signals(prices)
        if self.vol_target is not None:
            returns = compute_returns(prices)
            signals = volatility_target_sizing(
                signals, returns, target_vol=self.vol_target,
                vol_lookback=self.vol_lookback,
            )
        return signals


class TSMOMStrategy(Strategy):
    """
    Time-series momentum strategy.

    Goes long when the cumulative return over the look-back period is positive,
    and short when it is negative.  (Eq. 71)

    Parameters
    ----------
    lookback_k : int
        Look-back period for cumulative return (default: 252 ≈ 12 months).
    vol_target : float, optional
        If set, applies volatility targeting (Eq. 70).
    vol_lookback : int
        Look-back for volatility estimation.
    max_leverage : float
        Maximum absolute position size.
    """

    def __init__(
        self,
        lookback_k: int = 252,
        vol_target: Optional[float] = 0.15,
        vol_lookback: int = 60,
        max_leverage: float = 2.0,
        name: str = "TSMOM",
    ):
        super().__init__(name=name)
        self.lookback_k = lookback_k
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        returns = compute_returns(prices)
        return tsmom_signal(returns, lookback_k=self.lookback_k)

    def get_positions(self, prices: pd.Series, **kwargs) -> pd.Series:
        returns = compute_returns(prices)
        signals = tsmom_signal(returns, lookback_k=self.lookback_k)

        if self.vol_target is not None:
            positions = volatility_target_sizing(
                signals, returns, target_vol=self.vol_target,
                vol_lookback=self.vol_lookback,
            )
        else:
            positions = signals

        return apply_leverage_cap(positions, max_leverage=self.max_leverage)


class SMACrossoverStrategy(Strategy):
    """
    Simple moving-average crossover strategy.  (Eq. 72)

    Goes long when the short SMA is above the long SMA, and short otherwise.

    Parameters
    ----------
    short_window : int
        Short SMA period K1 (default: 50).
    long_window : int
        Long SMA period K2 (default: 200).
    """

    def __init__(
        self,
        short_window: int = 50,
        long_window: int = 200,
        vol_target: Optional[float] = 0.15,
        vol_lookback: int = 60,
        max_leverage: float = 2.0,
        name: str = "SMA Crossover",
    ):
        super().__init__(name=name)
        self.short_window = short_window
        self.long_window = long_window
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        raw = sma_crossover_signal(
            prices, short_window=self.short_window,
            long_window=self.long_window,
        )
        return np.sign(raw)

    def get_positions(self, prices: pd.Series, **kwargs) -> pd.Series:
        signals = self.generate_signals(prices)
        if self.vol_target is not None:
            returns = compute_returns(prices)
            positions = volatility_target_sizing(
                signals, returns, target_vol=self.vol_target,
                vol_lookback=self.vol_lookback,
            )
        else:
            positions = signals
        return apply_leverage_cap(positions, max_leverage=self.max_leverage)


class MACDStrategy(Strategy):
    """
    MACD trend-following strategy.  (Eqs. 73–74)

    Can operate with a single MACD signal or combine multiple time-scales.

    Parameters
    ----------
    pairs : list of (short_span, long_span), optional
        If provided, uses the combined multi-scale MACD (Eq. 74).
        Default: ``[(8, 24), (16, 48), (32, 96)]``.
    short_span : int
        Short EWMA span for single-pair mode.
    long_span : int
        Long EWMA span for single-pair mode.
    use_combined : bool
        If True, use combined multi-scale MACD.
    """

    def __init__(
        self,
        use_combined: bool = True,
        pairs: Optional[List[Tuple[int, int]]] = None,
        short_span: int = 12,
        long_span: int = 26,
        vol_target: Optional[float] = 0.15,
        vol_lookback: int = 60,
        max_leverage: float = 2.0,
        name: str = "MACD",
    ):
        super().__init__(name=name)
        self.use_combined = use_combined
        self.pairs = pairs
        self.short_span = short_span
        self.long_span = long_span
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage

    def generate_signals(self, prices: pd.Series, **kwargs) -> pd.Series:
        if self.use_combined:
            raw = combined_macd_signal(prices, pairs=self.pairs)
        else:
            raw = macd_signal(
                prices, short_span=self.short_span, long_span=self.long_span,
            )
        return np.sign(raw)

    def get_positions(self, prices: pd.Series, **kwargs) -> pd.Series:
        signals = self.generate_signals(prices)
        if self.vol_target is not None:
            returns = compute_returns(prices)
            positions = volatility_target_sizing(
                signals, returns, target_vol=self.vol_target,
                vol_lookback=self.vol_lookback,
            )
        else:
            positions = signals
        return apply_leverage_cap(positions, max_leverage=self.max_leverage)


class CarryStrategy(Strategy):
    """
    FX carry trade strategy.  (Eqs. 75–76)

    Goes long the higher-yielding currency and short the lower-yielding one,
    exploiting the interest rate differential (IRD).

    Parameters
    ----------
    vol_target : float, optional
        If set, applies volatility targeting.
    leverage : float
        Leverage multiplier *l* in Eq. (76).
    """

    def __init__(
        self,
        vol_target: Optional[float] = 0.15,
        vol_lookback: int = 60,
        leverage: float = 1.0,
        max_leverage: float = 2.0,
        name: str = "FX Carry",
    ):
        super().__init__(name=name)
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.leverage = leverage
        self.max_leverage = max_leverage

    def generate_signals(
        self,
        prices: pd.Series,
        rate_a: Optional[pd.Series] = None,
        rate_b: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Generate carry signal from interest rate series.

        Parameters
        ----------
        prices : pd.Series
            FX spot price series (used for index alignment).
        rate_a : pd.Series
            Interest rate of the long currency.
        rate_b : pd.Series
            Interest rate of the short (funding) currency.
        """
        if rate_a is None or rate_b is None:
            raise ValueError(
                "CarryStrategy requires `rate_a` and `rate_b` interest rate "
                "series passed as keyword arguments."
            )
        sig = carry_signal(rate_a, rate_b)
        return sig.reindex(prices.index, method="ffill").fillna(0)

    def get_positions(
        self,
        prices: pd.Series,
        rate_a: Optional[pd.Series] = None,
        rate_b: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        signals = self.generate_signals(prices, rate_a=rate_a, rate_b=rate_b)
        signals = signals * self.leverage

        if self.vol_target is not None:
            returns = compute_returns(prices)
            positions = volatility_target_sizing(
                signals, returns, target_vol=self.vol_target,
                vol_lookback=self.vol_lookback,
            )
        else:
            positions = signals

        return apply_leverage_cap(positions, max_leverage=self.max_leverage)

    def compute_daily_interest(
        self,
        rate_a: pd.Series,
        rate_b: pd.Series,
        notional: float,
        leverage: float = 1.0,
    ) -> pd.Series:
        """
        Compute net daily interest earned from carry trade.  (Eq. 76)

        .. math::
            I = (i_A - i_B) \\times C \\times l / 365

        Parameters
        ----------
        rate_a : pd.Series
            Annualized interest rate of long currency.
        rate_b : pd.Series
            Annualized interest rate of short currency.
        notional : float
            Notional capital *C*.
        leverage : float
            Leverage multiplier *l*.

        Returns
        -------
        pd.Series
            Daily interest income.
        """
        ird = rate_a - rate_b
        return ird * notional * leverage / 365
