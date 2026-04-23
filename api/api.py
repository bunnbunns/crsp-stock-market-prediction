from datetime import datetime
import yfinance as yf
import numpy as np
from flask import Flask, request, jsonify
from tensorflow import keras
import ml_helpers as ml
from itertools import combinations

app = Flask(__name__)

num_features = 4

def filter_stocks(x):
	if len(x) == num_features:
		return True
	return False

@app.route('/ml/train/<ticker>/<start>', methods=['POST'])
def train(ticker, start, end = datetime.today().strftime('%Y-%m-%d')):
	ticker_data = yf.Ticker(ticker)
	data = request.get_json()
	features = data.get('indicators', [])

	if not ticker_data.info or 'symbol' not in ticker_data.info:
			return "Error: {ticker} is not a valid ticker."
	
	try:
		tups = []
		# for r in range(len(features) + 1):
		# 	tups.extend(combinations(features, r))
		
		# all_combos = list(map(list, tups))
		# feature_combos = list(filter(filter_stocks, all_combos))
		feature_combos = list(combinations(features, num_features))
		result = []

		model = keras.models.Sequential([
			keras.layers.Input((None, num_features + 1)),
			# Use fewer units and add Dropout to prevent "lazy" learning
			keras.layers.LSTM(128, return_sequences=False), 
			keras.layers.Dropout(0.2),
			keras.layers.Dense(64, activation='relu'),
			keras.layers.Dense(1)
		])
		optimizer = keras.optimizers.Adam(learning_rate=0.001)
		model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
		
		counter = 1
		for feature in feature_combos:
			result_dict = {}
			model_results = ml.train_model(ticker_data, feature, start, end, model)
			result_dict = {
        'features': feature,
        'accuracy': model_results['accuracy'],
        'predictions': model_results['predictions']
    	}
			result.append(result_dict)
			print(len(result), '/', len(feature_combos))

		result.sort(key=lambda x: x['accuracy'], reverse=True)

		return jsonify(result), 200

	except Exception as e:
		print(e)

	return "there was an error"