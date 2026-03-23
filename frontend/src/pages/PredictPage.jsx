import { useState, useEffect, useRef } from 'react'
import { getYears, getEvents, predictLiveStream } from '../api'

const positionBadge = (pos) => {
  if (pos === 1) return 'bg-yellow-500 text-black'
  if (pos === 2) return 'bg-gray-300 text-black'
  if (pos === 3) return 'bg-amber-700 text-white'
  return 'bg-zinc-700 text-white'
}

const formatFeatureVal = (v) => {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return v.toString()
    return v.toFixed(3)
  }
  return String(v)
}

function FeaturesTable({ result }) {
  const drivers = result.drivers
  const featureNames = result.feature_names || []
  if (!featureNames.length || !drivers.length) return null

  return (
    <div className="mt-8">
      <h3 className="text-base font-semibold text-white mb-3">Engineered Features</h3>
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="text-xs whitespace-nowrap">
          <thead>
            <tr className="bg-zinc-800/80 text-zinc-400 uppercase tracking-wide">
              <th className="sticky left-0 z-10 bg-zinc-800 text-left px-4 py-2 min-w-64 font-medium">
                Feature
              </th>
              {drivers.map((d) => (
                <th key={d.abbreviation} className="text-center px-3 py-2 font-medium">
                  <span className="text-white font-mono">{d.abbreviation}</span>
                  <span className="block text-zinc-500 normal-case text-xs">
                    P{d.predicted_position}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {featureNames.map((feat, i) => (
              <tr
                key={feat}
                className={`border-t border-zinc-800 ${
                  i % 2 === 0 ? 'bg-zinc-900' : 'bg-zinc-900/50'
                }`}
              >
                <td className="sticky left-0 z-10 bg-inherit px-4 py-1.5 text-zinc-300 font-mono">
                  {feat}
                </td>
                {drivers.map((d) => {
                  const val = d.features?.[feat]
                  return (
                    <td
                      key={d.abbreviation}
                      className={`text-center px-3 py-1.5 ${
                        val === null || val === undefined ? 'text-zinc-600' : 'text-zinc-200'
                      }`}
                    >
                      {formatFeatureVal(val)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const DNF_STORAGE_KEY = 'f1_dnf_drivers'

function loadStoredDnfDrivers() {
  try { return JSON.parse(localStorage.getItem(DNF_STORAGE_KEY) || '[]') } catch { return [] }
}

function DnfSelect({ drivers, dnfDrivers, onAdd, onRemove }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const toggle = (abbr) => {
    if (dnfDrivers.includes(abbr)) onRemove(abbr)
    else onAdd(abbr)
  }

  const disabled = !drivers.length
  const label = dnfDrivers.length ? dnfDrivers.join(', ') : 'None'

  return (
    <div className="flex flex-col gap-1" ref={ref}>
      <label className="text-xs text-zinc-400 font-medium uppercase tracking-wide">
        Known DNFs
      </label>
      <div className="relative">
        <button
          type="button"
          disabled={disabled}
          onClick={() => setOpen((v) => !v)}
          className="flex items-center justify-between gap-2 bg-zinc-800 border border-zinc-700 text-sm rounded px-3 py-2 min-w-48 w-48 focus:outline-none focus:border-red-500 disabled:opacity-40 disabled:cursor-not-allowed hover:border-zinc-500 transition-colors"
        >
          <span className={`truncate ${dnfDrivers.length ? 'text-red-300 font-mono' : 'text-zinc-500'}`}>
            {disabled ? 'Run prediction first' : label}
          </span>
          <svg className="w-4 h-4 text-zinc-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {open && !disabled && (
          <div className="absolute top-full left-0 mt-1 w-56 bg-zinc-800 border border-zinc-700 rounded shadow-xl z-50 max-h-64 overflow-y-auto">
            {drivers.map((d) => {
              const checked = dnfDrivers.includes(d.abbreviation)
              return (
                <button
                  key={d.abbreviation}
                  type="button"
                  onClick={() => toggle(d.abbreviation)}
                  className={`w-full flex items-center gap-3 px-3 py-2 text-sm text-left hover:bg-zinc-700 transition-colors ${checked ? 'text-red-300' : 'text-white'}`}
                >
                  <span className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 ${checked ? 'bg-red-600 border-red-500' : 'border-zinc-600'}`}>
                    {checked && (
                      <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </span>
                  <span className="font-mono font-semibold">{d.abbreviation}</span>
                  {d.full_name && <span className="text-zinc-400 text-xs truncate">{d.full_name}</span>}
                </button>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

export default function PredictPage() {
  useEffect(() => { document.title = 'Predict' }, [])

  const [years, setYears] = useState([])
  const [events, setEvents] = useState([])
  const [selectedYear, setSelectedYear] = useState('')
  const [selectedEvent, setSelectedEvent] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showFeatures, setShowFeatures] = useState(false)
  const [progressMessages, setProgressMessages] = useState([])
  const [dnfDrivers, setDnfDrivers] = useState(loadStoredDnfDrivers)

  useEffect(() => {
    getYears().then((y) => {
      const sorted = [...y].sort((a, b) => b - a)
      setYears(sorted)
      if (sorted.length) setSelectedYear(sorted[0])
    })
  }, [])

  const clearDnf = () => {
    setDnfDrivers([])
    localStorage.removeItem(DNF_STORAGE_KEY)
  }

  useEffect(() => {
    if (!selectedYear) return
    setEvents([])
    setSelectedEvent('')
    setError(null)
    setResult(null)
    clearDnf()
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

  useEffect(() => {
    if (!selectedEvent) return
    setResult(null)
    clearDnf()
  }, [selectedEvent])

  const addDnf = (abbr) => {
    const updated = [...dnfDrivers, abbr]
    setDnfDrivers(updated)
    localStorage.setItem(DNF_STORAGE_KEY, JSON.stringify(updated))
  }

  const removeDnf = (abbr) => {
    const updated = dnfDrivers.filter((d) => d !== abbr)
    setDnfDrivers(updated)
    localStorage.setItem(DNF_STORAGE_KEY, JSON.stringify(updated))
  }

  const handlePredict = async () => {
    if (!selectedYear || !selectedEvent) return
    setLoading(true)
    setError(null)
    setResult(null)
    setProgressMessages([])
    try {
      const data = await predictLiveStream(selectedYear, selectedEvent, msg => {
        console.log('[F1 Predictor]', msg)
        setProgressMessages(prev => [...prev, msg])
      }, dnfDrivers)
      setResult(data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Prediction failed.')
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

        <DnfSelect
          drivers={result?.drivers?.map((d) => ({ abbreviation: d.abbreviation, full_name: d.full_name })) ?? []}
          dnfDrivers={dnfDrivers}
          onAdd={addDnf}
          onRemove={removeDnf}
        />

        <button
          onClick={handlePredict}
          disabled={loading || !selectedEvent}
          className="bg-red-600 hover:bg-red-500 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white font-semibold px-6 py-2 rounded text-sm transition-colors"
        >
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-950 border border-red-800 text-red-300 rounded px-4 py-3 text-sm mb-6">
          {error}
        </div>
      )}

      {/* Live prediction progress */}
      {loading && progressMessages.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-700 rounded px-4 py-3 text-sm mb-6 font-mono">
          {progressMessages.map((msg, i) => (
            <div
              key={i}
              className={`flex items-center gap-2 py-0.5 ${i === progressMessages.length - 1 ? 'text-white' : 'text-zinc-500'}`}
            >
              <span className="text-zinc-600 select-none">›</span>
              {i === progressMessages.length - 1 ? (
                <span className="flex items-center gap-2">
                  {msg}
                  <span className="inline-flex gap-0.5">
                    <span className="w-1 h-1 bg-zinc-400 rounded-full animate-bounce [animation-delay:0ms]" />
                    <span className="w-1 h-1 bg-zinc-400 rounded-full animate-bounce [animation-delay:150ms]" />
                    <span className="w-1 h-1 bg-zinc-400 rounded-full animate-bounce [animation-delay:300ms]" />
                  </span>
                </span>
              ) : msg}
            </div>
          ))}
        </div>
      )}

      {/* Results */}
      {result && (
        <div>
          {result.year >= 2026 && (
            <div className="bg-green-950 border border-green-700 text-green-300 rounded px-4 py-3 text-sm mb-4 flex gap-2 items-start">
              <span className="font-bold shrink-0">Live data:</span>
              <span>
                Predictions for {result.year} use qualifying and practice session data fetched
                directly from the FastF1 API.{' '}
                {result.training_cutoff ? (
                  result.training_cutoff.year >= result.year ? (
                    <>Model trained through <strong>{result.training_cutoff.event} {result.training_cutoff.year}</strong> (Round {result.training_cutoff.round}).</>
                  ) : (
                    <>Model trained on 2018–{result.training_cutoff.year} data only — no {result.year} races in training.</>
                  )
                ) : (
                  <>Training data: 2018–2025.</>
                )}
              </span>
            </div>
          )}
          {result.year < 2026 && (result.training_cutoff ? result.training_cutoff.year >= result.year : result.in_sample) && (
            <div className="bg-yellow-950 border border-yellow-700 text-yellow-300 rounded px-4 py-3 text-sm mb-4 flex gap-2 items-start">
              <span className="font-bold shrink-0">Warning:</span>
              <span>
                {result.year} was part of the model&apos;s training data (2018–{result.training_cutoff?.year ?? 2025}).
                These predictions are <strong>in-sample</strong> — the model has already seen this data
                and will appear more accurate than on unseen races. Only 2026+ predictions are a fair evaluation.
              </span>
            </div>
          )}
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">
              {result.year} {result.event}
            </h2>
            <div className="flex items-center gap-4">
              {result.median_error !== null && result.median_error !== undefined && (
                <span className="text-sm text-zinc-400">
                  Median Error: <span className="text-white font-medium">{result.median_error} positions</span>
                </span>
              )}
              {result.feature_names?.length > 0 && (
                <button
                  onClick={() => setShowFeatures((v) => !v)}
                  className="text-xs bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 px-3 py-1.5 rounded transition-colors"
                >
                  {showFeatures ? 'Hide Features' : 'Show Features'}
                </button>
              )}
            </div>
          </div>

          <div className="overflow-x-auto rounded-lg border border-zinc-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-zinc-800/60 text-zinc-400 uppercase text-xs tracking-wide">
                  <th className="text-left px-4 py-3">Pred</th>
                  <th className="text-left px-4 py-3">Driver</th>
                  <th className="text-left px-4 py-3">Team</th>
                  <th className="text-center px-4 py-3">Starting Position</th>
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
                    className={`border-t border-zinc-800 transition-colors ${
                      d.status === 'DNF'
                        ? 'bg-red-950/20 hover:bg-red-950/30 opacity-60'
                        : i % 2 === 0 ? 'bg-zinc-900 hover:bg-zinc-800/60' : 'bg-zinc-900/50 hover:bg-zinc-800/60'
                    }`}
                  >
                    <td className="px-4 py-3">
                      {d.status === 'DNF' ? (
                        <span className="inline-flex items-center justify-center px-2 h-7 rounded text-xs font-bold bg-red-900 border border-red-700 text-red-300">
                          DNF
                        </span>
                      ) : (
                        <span
                          className={`inline-flex items-center justify-center w-7 h-7 rounded text-xs font-bold ${positionBadge(
                            d.predicted_position
                          )}`}
                        >
                          {d.predicted_position}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`font-mono font-semibold ${d.status === 'DNF' ? 'text-zinc-500' : 'text-white'}`}>{d.abbreviation}</span>
                      {d.full_name && (
                        <span className="text-zinc-500 ml-2 text-xs">{d.full_name}</span>
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

          {showFeatures && <FeaturesTable result={result} />}
        </div>
      )}
    </div>
  )
}
