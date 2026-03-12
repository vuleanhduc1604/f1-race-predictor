import { NavLink } from 'react-router-dom'

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    `px-4 py-2 rounded text-sm font-medium transition-colors ${
      isActive
        ? 'bg-red-600 text-white'
        : 'text-gray-300 hover:text-white hover:bg-white/10'
    }`

  return (
    <nav className="bg-zinc-900 border-b border-zinc-800 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-4 flex items-center gap-6 h-14">
        <div className="flex items-center gap-2 mr-4">
          <img src="/favicon.svg" alt="F1" className="w-7 h-7" />
          <span className="text-white font-semibold text-sm">Race Predictor</span>
        </div>
        <NavLink to="/" className={linkClass} end>Predict</NavLink>
        <NavLink to="/features" className={linkClass}>Feature Importance</NavLink>
      </div>
    </nav>
  )
}
