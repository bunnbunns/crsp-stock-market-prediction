import numpy as np
import pandas as pd

""" Relative Strength Index (RSI) """
# df, number of days
def rsi(data, n = 14):
    delta   = data.Close.diff()

    up_days = delta.copy()
    up_days[delta <= 0] = 0.0

    down_days = delta.copy()
    down_days[delta > 0] = 0.0
    down_days = abs(down_days)

    RS_up = up_days.rolling(n).mean()
    RS_down = down_days.rolling(n).mean()

    rsi = 100 - 100 / (1 + RS_up / RS_down)

    return rsi.values

""" Moving Average (MA) """
# df, number of days
# adds MA column to the dataframe
def ma(data, n = 14):
    return data.Close.rolling(n).mean()

""" Exponential Moving Average (EMA) """
# df, number of days
# returns ema column
def ema(data, n = 50):
    k = 2/(n+2)
    return data.Close.ewm(alpha=k).mean()

# for use for other techinical indicator functions
def ewm(col, n = 50):
    k = 2/(n+2)
    return col.ewm(alpha=k).mean()

"""Moving Average Convergence - Divergence (MACD)"""
# will use the histogram for machine learning
# the typical emas to use are 12-day and 26-day
# n_f = fast EMA, n_s = slow EMA, n_m=signal EMA
def macd(data, n_f=12, n_s=26): # macd
    return np.subtract(ewm(data.Close, n_s), ewm(data.Close, n_f))

def ma_macd(data, n_f=12, n_s=26, n_m=9): # signal line
    return ewm(macd(data, n_f, n_s), n_m)

def diff_macd(data, n_f=12, n_s=26, n_m=9): # histogram
    return np.subtract(macd(data, n_f, n_s), ma_macd(data, n_f, n_s, n_m))

"""Percentage Price Oscillator"""
def ppo(data, n_f=12, n_s=26):
    return np.divide(np.subtract(ewm(data.Close, n_s), ewm(data.Close, n_f)), ewm(data.Close, n_f)) * 100

def ma_ppo(data, n_f=12, n_s=26, n_m=9):
    return ewm(ppo(data, n_f, n_s), n_m)

def diff_ppo(data, n_f=12, n_s=26, n_m=9): # histogram
    return np.subtract(ppo(data, n_f, n_s), ma_ppo(data, n_f, n_s, n_m))

tech_dict = {
  'RSI': rsi,
  'MA': ma,
  'EMA': ema,
  'MACD': diff_macd,
  'PPO': diff_ppo
}