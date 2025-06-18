import { useState } from 'react'
import { Calendar, TrendingUp, BarChart3, PieChart, Activity } from 'lucide-react'

const Analytics = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('1M')
  const [selectedMetric, setSelectedMetric] = useState('returns')

  const periods = [
    { value: '1D', label: '1 Day' },
    { value: '1W', label: '1 Week' },
    { value: '1M', label: '1 Month' },
    { value: '3M', label: '3 Months' },
    { value: '6M', label: '6 Months' },
    { value: '1Y', label: '1 Year' },
    { value: 'YTD', label: 'YTD' },
    { value: 'ALL', label: 'All Time' },
  ]

  const metrics = [
    { id: 'returns', label: 'Returns', icon: TrendingUp },
    { id: 'allocation', label: 'Allocation', icon: PieChart },
    { id: 'performance', label: 'Performance', icon: BarChart3 },
    { id: 'risk', label: 'Risk Analysis', icon: Activity },
  ]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Analytics</h1>
          <p className="text-gray-400 mt-1">Deep insights into your portfolio performance</p>
        </div>
        
        <button className="btn-secondary flex items-center space-x-2">
          <Calendar className="w-5 h-5" />
          <span>Custom Range</span>
        </button>
      </div>

      {/* Period Selector */}
      <div className="flex items-center space-x-2 bg-dark-surface rounded-lg p-1 w-fit">
        {periods.map((period) => (
          <button
            key={period.value}
            onClick={() => setSelectedPeriod(period.value)}
            className={`px-4 py-2 text-sm font-medium rounded transition-colors ${
              selectedPeriod === period.value
                ? 'bg-primary-600 text-white'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {period.label}
          </button>
        ))}
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric) => (
          <button
            key={metric.id}
            onClick={() => setSelectedMetric(metric.id)}
            className={`card flex flex-col items-center justify-center py-8 transition-all duration-200 ${
              selectedMetric === metric.id
                ? 'border-primary-500 bg-primary-600/10'
                : 'hover:border-gray-600'
            }`}
          >
            <metric.icon className={`w-8 h-8 mb-3 ${
              selectedMetric === metric.id ? 'text-primary-400' : 'text-gray-400'
            }`} />
            <span className={`font-medium ${
              selectedMetric === metric.id ? 'text-primary-400' : 'text-gray-300'
            }`}>
              {metric.label}
            </span>
          </button>
        ))}
      </div>

      {/* Analytics Content */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-100 mb-6">
          {metrics.find(m => m.id === selectedMetric)?.label} Analysis
        </h2>
        <div className="h-96 flex items-center justify-center text-gray-400">
          <p>Analytics visualization for {selectedMetric} will be displayed here</p>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Performance Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Sharpe Ratio</span>
              <span className="font-medium text-gray-100">1.45</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Alpha</span>
              <span className="font-medium text-success">+2.3%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Beta</span>
              <span className="font-medium text-gray-100">0.85</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Risk Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Volatility</span>
              <span className="font-medium text-gray-100">12.5%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Drawdown</span>
              <span className="font-medium text-error">-8.2%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">VaR (95%)</span>
              <span className="font-medium text-gray-100">$2,450</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-100 mb-4">Return Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Return</span>
              <span className="font-medium text-success">+18.7%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Annualized Return</span>
              <span className="font-medium text-success">+22.4%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Win Rate</span>
              <span className="font-medium text-gray-100">65%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics