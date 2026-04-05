import numpy as np
import pandas as pd
from technical_indicators import tech_dict

## Lag Function
def lag_func(data, name, lag):
  df_lag = pd.DataFrame(data[name])
  for i in range(1, lag+1):
    df_lag[f'lag_{i}'] = df_lag[name].shift(i)
    df_lag.dropna(inplace=True)
  return df_lag

## Add Technical Indicators to Dataframe
def add_ti_to_df(data, ti_arr):
	ti_df = data[['Close']].copy()

	for ti in ti_arr:
		ti_df[ti] = tech_dict[ti](data)
  
	ratio = data.Close / data.Close.shift(1)
	ti_df.Close = np.log(ratio.where(ratio > 0))
	ti_df.dropna(inplace=True)
	return ti_df

## Make Dictionary of Lagged Technical Indicator Data
def add_to_dict(data):
  df_dict = {}
  for col in data.columns:
    df_dict[col] =  lag_func(data, name=col, lag=10)
  return df_dict

## Generate Training Data
def generate_training_data(dict_data, df_data):
	yR = dict_data['Close'].Close.values
	yC = np.where(yR > 0, 1, 0)
	X = dict_data['Close'].iloc[:,1:].values
	X = X.reshape(X.shape+(1,))

	for col in df_data.columns[1:]:
		new_input = dict_data[col].iloc[:,1:].values
		new_input = new_input.reshape(new_input.shape+(1,))
		X = np.concatenate([X, new_input], axis=2)

	N = X.shape[0] # total number of rows

	tr = 0.90 # train ratio
	vr = (1-tr)/2 # validation ratio

	ts = int(N*tr) # training size
	vs = int(N*vr) # validation size

	X_train, yR_train, yC_train = X[:ts], yR[:ts], yC[:ts]
	X_valid, yR_valid, yC_valid = X[ts:ts+vs], yR[ts:ts+vs], yC[ts:ts+vs]
	X_test , yR_test , yC_test  = X[ts+vs:], yR[ts+vs:], yC[ts+vs:]

	train = X_train, yR_train, yC_train
	valid = X_valid, yR_valid, yC_valid
	test  = X_test , yR_test , yC_test

	return train, valid, test

