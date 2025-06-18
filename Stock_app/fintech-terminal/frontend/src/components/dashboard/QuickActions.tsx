import { useState } from 'react'
import { Plus, Search, Bell, Download, Upload, BarChart3 } from 'lucide-react'

const QuickActions = () => {
  const [searchSymbol, setSearchSymbol] = useState('')

  const actions = [
    {
      icon: Plus,
      label: 'Buy/Sell',
      color: 'bg-primary-600 hover:bg-primary-700',
      onClick: () => console.log('Open trade modal'),
    },
    {
      icon: Bell,
      label: 'Set Alert',
      color: 'bg-warning/20 hover:bg-warning/30 text-warning',
      onClick: () => console.log('Open alert modal'),
    },
    {
      icon: Download,
      label: 'Export Data',
      color: 'bg-success/20 hover:bg-success/30 text-success',
      onClick: () => console.log('Export data'),
    },
    {
      icon: BarChart3,
      label: 'Analytics',
      color: 'bg-primary-600/20 hover:bg-primary-600/30 text-primary-400',
      onClick: () => console.log('Navigate to analytics'),
    },
  ]

  return (
    <div className="card">
      <h2 className="text-xl font-semibold text-gray-100 mb-6">Quick Actions</h2>

      {/* Symbol Search */}
      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={searchSymbol}
            onChange={(e) => setSearchSymbol(e.target.value)}
            placeholder="Quick symbol lookup..."
            className="input pl-10"
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-3">
        {actions.map((action, index) => (
          <button
            key={index}
            onClick={action.onClick}
            className={`flex flex-col items-center justify-center p-4 rounded-lg transition-all duration-200 ${action.color}`}
          >
            <action.icon className="w-6 h-6 mb-2" />
            <span className="text-sm font-medium">{action.label}</span>
          </button>
        ))}
      </div>

      {/* Market Status */}
      <div className="mt-6 p-4 bg-dark-surface rounded-lg">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Market Status</span>
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 bg-success rounded-full animate-pulse"></span>
            <span className="text-sm font-medium text-success">Open</span>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-1">Closes at 4:00 PM ET</p>
      </div>

      {/* Recent Searches */}
      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Recent Searches</h3>
        <div className="space-y-2">
          {['AAPL', 'TSLA', 'NVDA'].map((symbol) => (
            <button
              key={symbol}
              className="w-full text-left px-3 py-2 text-sm text-gray-300 bg-dark-surface rounded hover:bg-dark-border transition-colors"
            >
              {symbol}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default QuickActions