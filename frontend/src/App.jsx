import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import PredictPage from './pages/PredictPage'
import EvaluatePage from './pages/EvaluatePage'
import FeatureImportancePage from './pages/FeatureImportancePage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-zinc-950 text-white">
        <Navbar />
        <main>
          <Routes>
            <Route path="/" element={<PredictPage />} />
            <Route path="/evaluate" element={<EvaluatePage />} />
            <Route path="/features" element={<FeatureImportancePage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
