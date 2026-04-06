import { useState } from 'react'

const technicalIndicators = ['RSI', 'SMA', 'EMA', 'MACD', 'PPO', 'SO', 'SD'];

export function TickerForm() {
	const [accuracy, setAccuracy] = useState(0.0);
	const [techIndicators, setTechIndicators] = useState([]);

	const train = (formData) => {
		const ticker = formData.get("ticker");
		fetch(`/ml/train/${ticker}/2021-01-01`, {
			method: 'POST',
    	headers: { 'Content-Type': 'application/json' },
    	body: JSON.stringify({ indicators: techIndicators })
		})
			.then(res => res.json())
			.then(data => {
				setAccuracy(data);
		});
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
				<button>Enter</button>
			</form>

			<h3>Accuracy: {accuracy}</h3>
		</div>
	)
}
