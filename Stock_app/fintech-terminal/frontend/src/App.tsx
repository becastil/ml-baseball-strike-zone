import { Routes, Route } from 'react-router-dom'
import { Suspense, lazy } from 'react'
import Header from './components/layout/Header'
import Sidebar from './components/layout/Sidebar'

// Lazy load pages
const Dashboard = lazy(() => import('./pages/Dashboard'))
const MarketView = lazy(() => import('./pages/MarketView'))
const Portfolio = lazy(() => import('./pages/Portfolio'))
const Analytics = lazy(() => import('./pages/Analytics'))

function App() {
  return (
    <div className="flex h-screen bg-dark-bg text-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto bg-dark-bg p-6">
          <Suspense fallback={
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-500"></div>
            </div>
          }>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/market" element={<MarketView />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </div>
  )
}

export default App