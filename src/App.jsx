import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [accuracy, setAccuracy] = useState(0.0);

  const train = (formData) => {
    const ticker = formData.get("ticker");
    fetch(`/ml/train/${ticker}/2021-01-01`)
      .then(res => res.json())
      .then(data => {
        alert(data)
        setAccuracy(data);
    })
  }

  return (
    <>
      <form action={train}>
        <label>Enter Stock Ticker</label><br />
        <input name="ticker" />
        <button>Enter</button>
      </form>

      <h1>{accuracy}</h1>
    </>
  )
}

export default App
