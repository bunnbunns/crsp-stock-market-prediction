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
def sma(data, n = 14):
    return data.Close.rolling(n).mean()

""" Exponential Moving Average (EMA) """
# df, number of days
# returns ema column
def ema(data, n = 50):
    k = 2/(n+2)
    return data.Close.ewm(alpha=k).mean()

# df column, number of days
# for use for other techinical indicator functions
def ewm(col, n = 50):
    k = 2/(n+2)
    return col.ewm(alpha=k).mean()

"""Moving Average Convergence - Divergence (MACD)"""
# will use the histogram for machine learning
# the typical emas to use are 12-day and 26-day
# df, fast EMA, slow EMA
def macd(data, n_f=12, n_s=26): # macd
    return np.subtract(ewm(data.Close, n_s), ewm(data.Close, n_f))

# df, fast EMA, slow EMA, signal EMA
def ma_macd(data, n_f=12, n_s=26, n_m=9): # signal line
    return ewm(macd(data, n_f, n_s), n_m)

# df, fast EMA, slow EMA, signal EMA
def diff_macd(data, n_f=12, n_s=26, n_m=9): # histogram
    return np.subtract(macd(data, n_f, n_s), ma_macd(data, n_f, n_s, n_m))

"""Percentage Price Oscillator"""
def ppo(data, n_f=12, n_s=26):
    return np.divide(np.subtract(ewm(data.Close, n_s), ewm(data.Close, n_f)), ewm(data.Close, n_f)) * 100

def ma_ppo(data, n_f=12, n_s=26, n_m=9):
    return ewm(ppo(data, n_f, n_s), n_m)

def diff_ppo(data, n_f=12, n_s=26, n_m=9): # histogram
    return np.subtract(ppo(data, n_f, n_s), ma_ppo(data, n_f, n_s, n_m))

"""Stochastic Oscillator (SO)"""
# df, number of days
def so(data, n=14):
    df_temp = pd.DataFrame(data[['Close', 'High', 'Low']])
    df_temp['Low_N'] = df_temp.Low.rolling(n).min()
    df_temp['High_N'] = df_temp.High.rolling(n).max()
    return (data.Close-df_temp['Low_N']) / (df_temp['High_N']-df_temp['Low_N']) * 100

"""Standard Deviation (SD)"""
# df, number of days
def sd(data, n=20):
    return data.Close.rolling(n).std()

"""Commodity Channel Index (CCI)"""
# df, number of days
def cci(data, n=20):
    df_temp = pd.DataFrame(data[['Close', 'High', 'Low']])
    df_temp['TP'] = (df_temp['High'] + df_temp['Low'] + df_temp['Close']) / 3
    df_temp['SMA_TP'] = df_temp['TP'].rolling(n).mean()
    df_temp['MD']  = df_temp['TP'].rolling(n).apply(lambda x: abs(x - x.mean()).mean())
    return (df_temp['TP']-df_temp['SMA_TP']) / (0.015 * df_temp['MD'])

"""Rate of Change (ROC)"""
# df, number of days
def roc(data, n=12):
    return np.divide(np.subtract(data.Close, data.Close.shift(n)), data.Close.shift(n)) * 100

"""On-Balance Volume (OBV)"""
#df, number of days
def obv(data):
    diff = data['Close'].diff()
    signed_vol = np.sign(diff) * data['Volume']

    return signed_vol.cumsum()

"""Chaikin Money Flow (CMF)"""
#df, number of days
def cmf(data, n=21):
    mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    mfv = mfm * data['Volume']
    return mfv.rolling(n).sum() / data['Volume'].rolling(n).sum()

# ['RSI', 'SMA', 'EMA', 'MACD', 'PPO', 'SO', 'SD', 'CCI', 'ROC', 'OBV', 'CMF']

tech_dict = {
  'RSI': rsi,
  'SMA': sma,
  'EMA': ema,
  'MACD': diff_macd,
  'PPO': diff_ppo,
  'SO': so,
  'SD': sd,
  'CCI': cci,
  'ROC': roc,
  'OBV': obv,
  'CMF': cmf
}