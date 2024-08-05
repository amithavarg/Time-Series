import logging
from statsmodels.tsa.stattools import adfuller

def check_stationarity(data, alpha=0.05):
    try:
        result = adfuller(data)
        p_value = result[1]
        return p_value < alpha
    except Exception as e:
        logging.error(f"Error checking stationarity: {e}")
        raise

def difference(data, order=1):
    try:
        differenced = data.diff(periods=order).dropna()
        return differenced
    except Exception as e:
        logging.error(f"Error differencing series: {e}")
        raise
