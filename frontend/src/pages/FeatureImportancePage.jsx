import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { getFeatureImportance } from '../api'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm">
      <p className="text-white font-medium">{payload[0].payload.feature}</p>
      <p className="text-zinc-400">
        Importance: <span className="text-white">{payload[0].value}</span>
      </p>
    </div>
  )
}

export default function FeatureImportancePage() {
  useEffect(() => { document.title = 'Feature Importance' }, [])

  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [topN, setTopN] = useState(25)

  useEffect(() => {
    setLoading(true)
    setError(null)
    getFeatureImportance(topN)
      .then(setData)
      .catch((e) => setError(e.response?.data?.detail || 'Failed to load feature importance.'))
      .finally(() => setLoading(false))
  }, [topN])

  // Recharts needs data in ascending order for horizontal bar layout
  const chartData = [...data].reverse()

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-bold text-white mb-1">Feature Importance</h1>
      <p className="text-zinc-400 text-sm mb-6">
        LightGBM split-count importance for the top features used by the model.
      </p>

      <div className="flex items-center gap-3 mb-8">
        <label className="text-xs text-zinc-400 font-medium uppercase tracking-wide">
          Show top
        </label>
        {[15, 25, 40].map((n) => (
          <button
            key={n}
            onClick={() => setTopN(n)}
            className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
              topN === n
                ? 'bg-red-600 text-white'
                : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
            }`}
          >
            {n}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-950 border border-red-800 text-red-300 rounded px-4 py-3 text-sm mb-6">
          {error}
        </div>
      )}

      {loading && (
        <div className="text-zinc-400 text-sm">Loading…</div>
      )}

      {!loading && !error && data.length > 0 && (
        <div className="bg-zinc-800/40 rounded-lg border border-zinc-800 p-4">
          <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 24)}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 20, left: 20, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#a1a1aa', fontSize: 11 }}
                axisLine={{ stroke: '#52525b' }}
                tickLine={false}
                label={{ value: 'Split count', position: 'insideBottom', offset: -2, fill: '#71717a', fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="feature"
                width={210}
                tick={{ fill: '#d4d4d8', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
              <Bar dataKey="importance" fill="#3b82f6" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
