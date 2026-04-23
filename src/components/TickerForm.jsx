import { useState } from 'react'

const technicalIndicators = ['RSI', 'SMA', 'EMA', 'MACD', 'PPO', 'SO', 'SD', 'CCI', 'ROC', 'OBV', 'CMF'];

export function TickerForm() {
	const [accuracy, setAccuracy] = useState([]);
	const [techIndicators, setTechIndicators] = useState([]);

	const train = (formData, e) => {
		const ticker = formData.get("ticker");
		fetch(`/ml/train/${ticker}/2024-01-01`, {
			method: 'POST',
    	headers: { 'Content-Type': 'application/json' },
    	body: JSON.stringify({ indicators: techIndicators })
		})
			.then(res => res.json())
			.then(data => {
				setAccuracy(data)
		});
		console.log(e)
	}

	const handleIndicators = (indicator) => {
		setTechIndicators((prev) =>
			prev.includes(indicator)
				? prev.filter((i) => i !== indicator)
				: [...prev, indicator]
		);
	}
	
	return (
		<div>
			<form action={train}>
				<h3>Technical Indicators</h3>
				{technicalIndicators.map(indicator => (
					<div key={indicator}>
						<input type="checkbox" checked={techIndicators.includes(indicator)} onChange={() => handleIndicators(indicator)} />
						<label>{indicator}</label>
					</div>
				))}

				{}
				<h3>Enter Stock Ticker</h3><br />
				<input name="ticker" />
				<button type="submit">Enter</button>
			</form>

			<table>
				<thead>
					<tr>
						<th>Features</th>
						<th>Accuracy</th>
					</tr>
				</thead>
				<tbody>
					{accuracy.map((data, i) => (
						<tr key={`data${i}`}>
							<td>{data.features.map((feature, j) => (
								<span key={`feature${i}${j}`}>
									{j > 0 && ", "}
									{feature}
								</span>
							))}</td>
							<td>{data.accuracy}</td>
							<td>{data.predictions.map((prediction, j) => (
								<span key={`prediction${i}${j}`}>
									{j > 0 && ", "}
									{prediction[0]}
								</span>
							))}</td>
						</tr>
					))}
				</tbody>
			</table>
		</div>
	)
}
