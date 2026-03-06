import { useState, useEffect, useRef } from 'react'
import { getYears, getEvents, predict, predictLive } from '../api'

const STAGES = [
  { at: 5,  label: 'Fetching session data' },
  { at: 28, label: 'Processing lap times' },
  { at: 52, label: 'Generating features' },
  { at: 75, label: 'Running model' },
  { at: 91, label: 'Finalizing results' },
]

function stageLabel(p) {
  let label = STAGES[0].label
  for (const s of STAGES) {
    if (p >= s.at) label = s.label
  }
  return label
}

const positionBadge = (pos) => {
  if (pos === 1) return 'bg-yellow-500 text-black'
  if (pos === 2) return 'bg-gray-300 text-black'
  if (pos === 3) return 'bg-amber-700 text-white'
  return 'bg-zinc-700 text-white'
}

export default function PredictPage() {
  const [years, setYears] = useState([])
  const [events, setEvents] = useState([])
  const [selectedYear, setSelectedYear] = useState('')
  const [selectedEvent, setSelectedEvent] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(0)
  const progressTimer = useRef(null)

  useEffect(() => () => clearInterval(progressTimer.current), [])

  useEffect(() => {
    getYears().then((y) => {
      const sorted = [...y].sort((a, b) => b - a)
      setYears(sorted)
      if (sorted.length) setSelectedYear(sorted[0])
    })
  }, [])

  useEffect(() => {
    if (!selectedYear) return
    setEvents([])
    setSelectedEvent('')
    setError(null)
    setResult(null)
    getEvents(selectedYear)
      .then((e) => {
        setEvents(e)
        if (e.length) {
          setSelectedEvent(e[e.length - 1].name)
        } else {
          setError(`No events found for ${selectedYear}. The calendar may not be published yet.`)
        }
      })
      .catch((e) => {
        setError(
          e.response?.data?.detail ||
          `Could not load the ${selectedYear} race calendar. Check your connection.`
        )
      })
  }, [selectedYear])

  const handlePredict = async () => {
    if (!selectedYear || !selectedEvent) return
    setLoading(true)
    setError(null)
    setResult(null)
    setProgress(0)

    clearInterval(progressTimer.current)
    progressTimer.current = setInterval(() => {
      setProgress((prev) => {
        if (prev < 25) return prev + 0.8
        if (prev < 50) return prev + 0.4
        if (prev < 75) return prev + 0.2
        if (prev < 90) return prev + 0.1
        if (prev < 99) return prev + 0.04
        return prev
      })
    }, 80)

    try {
      const data = selectedYear >= 2025
        ? await predictLive(selectedYear, selectedEvent)
        : await predict(selectedYear, selectedEvent)
      clearInterval(progressTimer.current)
      setProgress(100)
      setResult(data)
    } catch (e) {
      clearInterval(progressTimer.current)
      setProgress(0)
      setError(e.response?.data?.detail || 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-white mb-1">Race Predictor</h1>
      <p className="text-zinc-400 text-sm mb-6">
        Select a season and race to generate predicted finishing positions.
      </p>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-end mb-8">
        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400 font-medium uppercase tracking-wide">
            Season
          </label>
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(Number(e.target.value))}
            className="bg-zinc-800 border border-zinc-700 text-white rounded px-3 py-2 text-sm focus:outline-none focus:border-red-500 min-w-28"
          >
            {years.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs text-zinc-400 font-medium uppercase tracking-wide">
            Race
          </label>
          <select
            value={selectedEvent}
            onChange={(e) => setSelectedEvent(e.target.value)}
            disabled={!events.length}
            className="bg-zinc-800 border border-zinc-700 text-white rounded px-3 py-2 text-sm focus:outline-none focus:border-red-500 min-w-64 disabled:opacity-50"
          >
            {events.map((ev) => (
              <option key={ev.name} value={ev.name}>
                R{ev.round} — {ev.name}
              </option>
            ))}
          </select>
        </div>

        <button
          onClick={handlePredict}
          disabled={loading || !selectedEvent}
          className="bg-red-600 hover:bg-red-500 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white font-semibold px-6 py-2 rounded text-sm transition-colors"
        >
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </div>

      {/* Progress bar */}
      {loading && (
        <div className="mb-6">
          <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-red-600 h-1.5 rounded-full transition-all duration-200 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-zinc-400 text-xs mt-2">{stageLabel(progress)}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-950 border border-red-800 text-red-300 rounded px-4 py-3 text-sm mb-6">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div>
          {result.source === 'live' && (
            <div className="bg-green-950 border border-green-700 text-green-300 rounded px-4 py-3 text-sm mb-4 flex gap-2 items-start">
              <span className="font-bold shrink-0">Live data:</span>
              <span>
                Predictions for {result.year} use qualifying and practice session data fetched
                directly from the FastF1 API. Race results are not used — no data leakage.
              </span>
            </div>
          )}
          {result.in_sample && (
            <div className="bg-yellow-950 border border-yellow-700 text-yellow-300 rounded px-4 py-3 text-sm mb-4 flex gap-2 items-start">
              <span className="font-bold shrink-0">Warning:</span>
              <span>
                {result.year} was part of the model&apos;s training data (2018–2024).
                These predictions are <strong>in-sample</strong> — the model has already seen this data
                and will appear more accurate than on unseen races. Only 2025 predictions are a fair evaluation.
              </span>
            </div>
          )}
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">
              {result.year} {result.event}
            </h2>
            {result.mae !== null && (
              <span className="text-sm text-zinc-400">
                MAE: <span className="text-white font-medium">{result.mae} positions</span>
              </span>
            )}
          </div>

          <div className="overflow-x-auto rounded-lg border border-zinc-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-zinc-800/60 text-zinc-400 uppercase text-xs tracking-wide">
                  <th className="text-left px-4 py-3">Pred</th>
                  <th className="text-left px-4 py-3">Driver</th>
                  <th className="text-left px-4 py-3">Team</th>
                  <th className="text-center px-4 py-3">Grid</th>
                  {result.has_actuals && (
                    <>
                      <th className="text-center px-4 py-3">Actual</th>
                      <th className="text-center px-4 py-3">Error</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {result.drivers.map((d, i) => (
                  <tr
                    key={d.abbreviation}
                    className={`border-t border-zinc-800 ${
                      i % 2 === 0 ? 'bg-zinc-900' : 'bg-zinc-900/50'
                    } hover:bg-zinc-800/60 transition-colors`}
                  >
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center justify-center w-7 h-7 rounded text-xs font-bold ${positionBadge(
                          d.predicted_position
                        )}`}
                      >
                        {d.predicted_position}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-mono font-semibold text-white">{d.abbreviation}</span>
                      {d.full_name && (
                        <span className="text-zinc-400 ml-2 text-xs">{d.full_name}</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-zinc-300 text-xs">{d.team || '—'}</td>
                    <td className="px-4 py-3 text-center text-zinc-300">{d.grid_position ?? '—'}</td>
                    {result.has_actuals && (
                      <>
                        <td className="px-4 py-3 text-center text-zinc-300">
                          {d.actual_position ?? '—'}
                        </td>
                        <td className="px-4 py-3 text-center">
                          <span
                            className={`text-xs font-medium ${
                              d.error === 0
                                ? 'text-green-400'
                                : d.error <= 2
                                ? 'text-yellow-400'
                                : 'text-red-400'
                            }`}
                          >
                            {d.error !== null ? (d.error === 0 ? '✓' : `±${d.error}`) : '—'}
                          </span>
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
