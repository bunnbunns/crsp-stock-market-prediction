from datetime import datetime
import yfinance as yf
import numpy as np
from flask import Flask
from flask import request
import ml_helpers as ml
from tensorflow import keras
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/ml/train/<ticker>/<START>', methods=['POST'])
def train(ticker, START, END = datetime.today().strftime('%Y-%m-%d')):
	ticker_data = yf.Ticker(ticker)
	data = request.get_json()
	features = data.get('indicators', [])
	print(features)
	df_dict = {}

	if not ticker_data.info or 'symbol' not in ticker_data.info:
			return "Error: {ticker} is not a valid ticker."
	
	try:
		# get stock data
		df = ticker_data.history(start=START, end=END)

		# remove time from dates
		df.reset_index(inplace=True)
		df['Date'] = df.Date.dt.date
		df.set_index('Date', inplace=True)
		df = df.iloc[:,:-2]

		df_ti = ml.add_ti_to_df(df, features)
		df_dict = ml.add_to_dict(df_ti)

		train, valid, test = ml.generate_training_data(df_dict, df_ti)
		X_train, yR_train, yC_train = train
		X_valid, yR_valid, yC_valid = valid
		X_test , yR_test , yC_test  = test

		model = keras.models.Sequential([
			keras.layers.Input((None, X_train.shape[2])),
			keras.layers.LSTM(100, activation='relu', return_sequences=True),
			keras.layers.LSTM(200, activation='relu'),
			keras.layers.Dense(1, activation='sigmoid')])
		model.compile(loss='binary_crossentropy', metrics=['accuracy'])

		model.fit(X_train, yC_train, validation_data=(X_valid, yC_valid))

		yC_test_pred = np.where(model.predict(X_test)>0.5, 1, 0)
		yC_test_pred[:5]

		return str(accuracy_score(yC_test, yC_test_pred))

	except Exception as e:
		print(e)

	return "boop"