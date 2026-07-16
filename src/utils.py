"""
src/utils.py
Shared utility functions - Gold Price Forecasting project
"""

import warnings
import pandas as pd

# Saudi Riyal has been pegged to USD at a fixed rate since June 1986
# (verified in EDA: SAR/USD ratio std is ~0.011 post-1986, i.e. effectively constant)
USD_TO_SAR_RATE = 3.75
SAR_PEG_START_DATE = pd.Timestamp("1986-06-01")


def usd_to_sar(value, dates=None):
    """
    Convert USD gold price value(s) to SAR using the fixed peg rate.

    Parameters
    ----------
    value : scalar, array-like, or pandas Series
        USD value(s) to convert.
    dates : array-like or pandas Series of dates, optional
        If provided, warns when any date falls before the peg start
        (1986-06-01), since the fixed 3.75 rate is not historically
        accurate before that point (SAR floated against a currency basket).
    """
    if dates is not None:
        dates = pd.to_datetime(pd.Series(dates))
        if (dates < SAR_PEG_START_DATE).any():
            warnings.warn(
                "Some dates are before the SAR/USD peg start (1986-06-01). "
                "The fixed 3.75 rate is not historically accurate for these rows."
            )
    return value * USD_TO_SAR_RATE