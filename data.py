"""
Futures contract data handling: nearest futures and continuous futures approaches
"""

import pandas as pd
import numpy as np


def compute_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    Compute returns from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series.
    method : str
        ``'simple'`` for arithmetic returns, ``'log'`` for log returns.

    Returns
    -------
    pd.Series
        Returns series (first value is NaN).
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    return prices.pct_change()


def build_nearest_futures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a price series using the nearest-futures approach.

    Selects the price of each contract until its expiration, then switches to the next contract. 
    This preserves actual historical prices but contains price gaps at rollovers 
    (not suitable for backtesting PnL).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'date', 'expire_date', 'future_contract', and 'price'.

    Returns
    -------
    pd.DataFrame
        Nearest-futures price and return series indexed by date.
    """
    if df.empty:
        return pd.DataFrame(dtype=float)

    # Filter out data points after the contract has expired
    valid_data = df[df['date'] <= df['expire_date']].copy()
    
    if valid_data.empty:
        return pd.DataFrame(dtype=float)

    # Sort by date and then by expiration date so the earliest expiring contract is first
    valid_data = valid_data.sort_values(by=['date', 'expire_date'])
    
    # Pick the nearest contract for each date
    nearest_futures = valid_data.drop_duplicates(subset=['date'], keep='first')
    nearest_futures = nearest_futures.set_index('date').sort_index()

    # Compute returns
    nearest_futures['return'] = compute_returns(nearest_futures['price'])

    return nearest_futures[['return', 'price']]


def build_continuous_futures(df: pd.DataFrame, method: str = "ratio") -> pd.Series:
    """
    Build a continuous (back-adjusted) futures price series.

    Eliminates price gaps at rollover points by adjusting historical prices, making the series suitable for backtesting.  
    The trade-off is that prices no longer match actual historical prices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'date', 'expire_date', 'future_contract', and 'price'.
    method : str
        Adjustment method: ``'ratio'`` (multiplicative) or ``'difference'`` (additive).

    Returns
    -------
    pd.Series
        Continuous, back-adjusted price series indexed by date.
    """
    if df.empty:
        return pd.Series(dtype=float)

    # Remove data points past the expiration date
    valid_data = df[df['date'] <= df['expire_date']].copy()
    if valid_data.empty:
        return pd.Series(dtype=float)

    # Find the active contract for each date (the one expiring next)
    valid_data = valid_data.sort_values(by=['date', 'expire_date'])
    active = valid_data.drop_duplicates(subset=['date'], keep='first').copy()
    active = active.set_index('date').sort_index()

    active_contract = active['future_contract']
    next_contract = active_contract.shift(-1)

    # Get the expiration dates of the contracts
    expiration_dates = active['expire_date'].unique()[:-1]

    # If there are no roll dates, return the price series as it is
    if len(expiration_dates) == 0:
        active['price_adjusted'] = active['price']
        active['return'] = compute_returns(active['price_adjusted'])
        return active[['return', 'price_adjusted', 'price']]

    # Pivot the raw dataframe so we can easily look up any contract price on any date
    all_prices = df.pivot(index='date', columns='future_contract', values='price')

    # Create a series to store the adjustment factors/differences at each expiration date
    adj_points = pd.Series(1.0 if method == "ratio" else 0.0, index=active.index)

    # Calculate the adjustment gap for each roll (expiration) date
    for e_date in expiration_dates:
        adj_points.loc[e_date] = \
            all_prices.loc[e_date, active_contract.loc[e_date]] \
            / all_prices.loc[e_date, next_contract.loc[e_date]]
        
    # Lag the adjustment points by one day so that the adjustment is applied 
    # to the price of the contract that is being replaced.
    adj_points = adj_points.shift(1)

    # Calculate cumulative product/sum
    if method == "ratio":
        cumulative_adj = adj_points.cumprod()
        active['price_adjusted'] = active['price'] * cumulative_adj
    else:
        cumulative_adj = adj_points.cumsum()
        active['price_adjusted'] = active['price'] + cumulative_adj
                
    # Compute returns
    active['return'] = compute_returns(active['price_adjusted'])

    return active[['return', 'price_adjusted', 'price']]

