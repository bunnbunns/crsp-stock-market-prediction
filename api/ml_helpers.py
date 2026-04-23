import numpy as np
import pandas as pd
from technical_indicators import tech_dict
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import gc

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
  
	ratio = data.Close / data.Close.shift(-1)
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
	yC = np.where(pd.Series(yR).shift(-1) > 0, 1, 0)
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

	# prep for scaling
	X_train_raw = X[:ts]
	X_valid_raw = X[ts:ts+vs]
	X_test_raw = X[ts+vs:]

	# scaler
	n_features = X.shape[2]
	scaler = MinMaxScaler(feature_range=(-1, 1))

	# Reshape to 2D for the scaler: (samples * time_steps, features)
	X_train_reshaped = X_train_raw.reshape(-1, n_features)
	X_valid_reshaped = X_valid_raw.reshape(-1, n_features)
	X_test_reshaped = X_test_raw.reshape(-1, n_features)

	# Fit only on Training data
	X_train_scaled = scaler.fit_transform(X_train_reshaped)

	# Transform Valid and Test based on Training fit
	X_valid_scaled = scaler.transform(X_valid_reshaped)
	X_test_scaled = scaler.transform(X_test_reshaped)

	X_train, yR_train, yC_train = X_train_scaled.reshape(X_train_raw.shape), yR[:ts], yC[:ts]
	X_valid, yR_valid, yC_valid = X_valid_scaled.reshape(X_valid_raw.shape), yR[ts:ts+vs], yC[ts:ts+vs]
	X_test , yR_test , yC_test  = X_test_scaled.reshape(X_test_raw.shape), yR[ts+vs:], yC[ts+vs:]

	train = X_train, yR_train, yC_train
	valid = X_valid, yR_valid, yC_valid
	test  = X_test , yR_test , yC_test

	return train, valid, test

def train_model(data, features, start, end, model):
	keras.backend.clear_session()
	gc.collect()
		# get stock data
	df = data.history(start=start, end=end)

	# remove time from dates
	df.reset_index(inplace=True)
	df['Date'] = df.Date.dt.date
	df.set_index('Date', inplace=True)
	df = df.iloc[:,:-2]

	# generate features
	df_ti = add_ti_to_df(df, features)
	df_dict = add_to_dict(df_ti)

	# generate training data
	train, valid, test = generate_training_data(df_dict, df_ti)
	X_train, yR_train, yC_train = train
	X_valid, yR_valid, yC_valid = valid
	X_test , yR_test , yC_test  = test

	# model
	# model = keras.models.Sequential([
	# 	keras.layers.Input((None, X_train.shape[2])),
	# 	# Use fewer units and add Dropout to prevent "lazy" learning
	# 	keras.layers.LSTM(64, return_sequences=False), 
	# 	keras.layers.Dropout(0.2),
	# 	keras.layers.Dense(32, activation='relu'),
	# 	keras.layers.Dense(1)
	# ])
	# optimizer = keras.optimizers.Adam(learning_rate=0.001)
	# model.compile(loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(X_train, yC_train, batch_size=32, validation_data=(X_valid, yC_valid), shuffle=False)

	yC_test_pred = np.where(model.predict(X_test)>0.5, 1, 0)

	result = {
		'accuracy': str(accuracy_score(yC_test, yC_test_pred)),
		'predictions': model.predict(X_test).tolist()
	}

	return result