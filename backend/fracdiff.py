import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import time

class FractionalDifferentiation:
    def __init__(self, tau: float = 1e-4):
        self.tau = tau
        self.d_params = {}
        
    def get_weights_ffd(self, d: float, tau: float) -> np.ndarray:
        """Finds the weights for FFD based on threshold tau."""
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < tau:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(self, series: pd.Series, d: float, tau: float = None) -> pd.Series:
        if tau is None:
            tau = self.tau
            
        w = self.get_weights_ffd(d, tau)
        width = len(w)
        
        series_filled = series.ffill().dropna()
        if len(series_filled) < width:
             return pd.Series(np.nan, index=series.index)
             
        w_flat = w.flatten()
        res = np.correlate(series_filled.values, w_flat, mode='valid')
        
        out = pd.Series(np.nan, index=series.index)
        out.loc[series_filled.index[width-1:]] = res
        return out

    def optimize_d(self, series: pd.Series, min_d: float = 0.3, max_d: float = 0.7, step: float = 0.05) -> float:
        for d in np.arange(min_d, max_d + 1e-5, step):
            fd = self.frac_diff_ffd(series, d)
            fd_clean = fd.dropna()
            if len(fd_clean) < 10:
                continue
            try:
                adf_stat = adfuller(fd_clean, maxlag=1, regression='c', autolag=None)
                p_val = adf_stat[1]
                if p_val < 0.05:
                    return d
            except:
                pass
        return max_d

    def fit(self, df_train: pd.DataFrame, value_col: str, ticker_col: str = 'ticker'):
        for ticker, group in df_train.groupby(ticker_col):
            series = np.log(group[value_col]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) > 0:
                d = self.optimize_d(series)
                self.d_params[ticker] = d
        return self

    def transform(self, df: pd.DataFrame, value_col: str, ticker_col: str = 'ticker') -> pd.Series:
        res = pd.Series(np.nan, index=df.index, dtype=float)
        for ticker, group in df.groupby(ticker_col):
            d = self.d_params.get(ticker, 0.5)
            series = np.log(group[value_col]).replace([np.inf, -np.inf], np.nan)
            res.loc[group.index] = self.frac_diff_ffd(series, d)
        return res
