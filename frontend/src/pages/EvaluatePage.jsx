import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell,
} from 'recharts'
import { getYears, evaluate } from '../api'

function StatCard({ label, value, unit = '' }) {
  return (
    <div className="bg-zinc-800 rounded-lg px-5 py-4">
      <p className="text-zinc-400 text-xs uppercase tracking-wide mb-1">{label}</p>
      <p className="text-white text-2xl font-bold">
        {value}
        {unit && <span className="text-zinc-400 text-sm font-normal ml-1">{unit}</span>}
      </p>
    </div>
  )
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm">
      <p className="text-white font-medium">{d.event.replace(' Grand Prix', ' GP')}</p>
      <p className="text-zinc-400">MAE: <span className="text-white">{d.mae}</span></p>
      <p className="text-zinc-400">Drivers: <span className="text-white">{d.drivers}</span></p>
    </div>
  )
}

export default function EvaluatePage() {
  useEffect(() => { document.title = 'Evaluate' }, [])

  const [years, setYears] = useState([])
  const [selectedYear, setSelectedYear] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getYears().then((y) => {
      const sorted = [...y].sort((a, b) => b - a)
      setYears(sorted)
      if (sorted.length) setSelectedYear(sorted[0])
    })
  }, [])

  const handleEvaluate = async () => {
    if (!selectedYear) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await evaluate(selectedYear)
      setResult(data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Evaluation failed.')
    } finally {
      setLoading(false)
    }
  }

  const meanMAE = result
    ? result.per_race.reduce((s, r) => s + r.mae, 0) / result.per_race.length
    : 0

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-white mb-1">Model Evaluation</h1>
      <p className="text-zinc-400 text-sm mb-6">
        Assess prediction accuracy across a full season.
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
        <button
          onClick={handleEvaluate}
          disabled={loading || !selectedYear}
          className="bg-red-600 hover:bg-red-500 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white font-semibold px-6 py-2 rounded text-sm transition-colors"
        >
          {loading ? 'Evaluating…' : 'Evaluate'}
        </button>
      </div>

      {error && (
        <div className="bg-red-950 border border-red-800 text-red-300 rounded px-4 py-3 text-sm mb-6">
          {error}
        </div>
      )}

      {result && (
        <>
          {selectedYear < 2025 && (
            <div className="bg-yellow-950 border border-yellow-700 text-yellow-300 rounded px-4 py-3 text-sm mb-6 flex gap-2 items-start">
              <span className="font-bold shrink-0">Warning:</span>
              <span>
                {selectedYear} was part of the model&apos;s training data (2018–2024).
                These metrics are <strong>in-sample</strong> — the model has already seen this data
                and will appear more accurate than the true out-of-sample performance.
                Only 2025 metrics reflect a fair evaluation.
              </span>
            </div>
          )}
          {/* Stat cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
            <StatCard label="Overall MAE" value={result.overall_mae} unit="pos" />
            <StatCard label="Median Error" value={result.median_error} unit="pos" />
            <StatCard label="RMSE" value={result.rmse} unit="pos" />
            <StatCard label="Within ±1" value={result.within_1} unit="%" />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-10">
            <StatCard label="Within ±2" value={result.within_2} unit="%" />
            <StatCard label="Within ±3" value={result.within_3} unit="%" />
            <StatCard label="Within ±5" value={result.within_5} unit="%" />
          </div>

          {/* Per-race bar chart */}
          <h2 className="text-lg font-semibold text-white mb-4">Per-Race MAE</h2>
          <div className="bg-zinc-800/40 rounded-lg border border-zinc-800 p-4">
            <ResponsiveContainer width="100%" height={Math.max(300, result.per_race.length * 22)}>
              <BarChart
                data={result.per_race}
                layout="vertical"
                margin={{ top: 4, right: 40, left: 10, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 'dataMax + 0.5']}
                  tick={{ fill: '#a1a1aa', fontSize: 11 }}
                  axisLine={{ stroke: '#52525b' }}
                  tickLine={false}
                  label={{ value: 'MAE (positions)', position: 'insideBottom', offset: -2, fill: '#71717a', fontSize: 11 }}
                />
                <YAxis
                  type="category"
                  dataKey="event"
                  width={170}
                  tick={{ fill: '#d4d4d8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => v.replace(' Grand Prix', ' GP')}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                <ReferenceLine
                  x={meanMAE}
                  stroke="#e10600"
                  strokeDasharray="4 3"
                  label={{ value: `Avg ${meanMAE.toFixed(2)}`, position: 'insideTopRight', fill: '#e10600', fontSize: 11 }}
                />
                <Bar dataKey="mae" radius={[0, 3, 3, 0]}>
                  {result.per_race.map((entry) => (
                    <Cell
                      key={entry.event}
                      fill={entry.mae > meanMAE ? '#e10600' : '#3b82f6'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}
