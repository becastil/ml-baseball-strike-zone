import { useState } from 'react'
import { useSelector } from 'react-redux'
import { RootState } from '@/store'
import { Plus, Download, TrendingUp, TrendingDown } from 'lucide-react'

const Portfolio = () => {
  const { activePortfolio, portfolios } = useSelector((state: RootState) => state.portfolio)
  const [selectedPortfolioId, setSelectedPortfolioId] = useState(activePortfolio?.id || '')

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Portfolio</h1>
          <p className="text-gray-400 mt-1">Manage your investments and track performance</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button className="btn-secondary flex items-center space-x-2">
            <Download className="w-5 h-5" />
            <span>Export</span>
          </button>
          <button className="btn-primary flex items-center space-x-2">
            <Plus className="w-5 h-5" />
            <span>Add Position</span>
          </button>
        </div>
      </div>

      {/* Portfolio Selector */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-100">Select Portfolio</h2>
          <select
            value={selectedPortfolioId}
            onChange={(e) => setSelectedPortfolioId(e.target.value)}
            className="input w-48"
          >
            {portfolios.map((portfolio) => (
              <option key={portfolio.id} value={portfolio.id}>
                {portfolio.name}
              </option>
            ))}
          </select>
        </div>

        {/* Portfolio Summary */}
        {activePortfolio && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-dark-surface rounded-lg p-4">
              <p className="text-sm text-gray-400">Total Value</p>
              <p className="text-2xl font-bold text-gray-100 mt-1">
                ${activePortfolio.totalValue.toLocaleString()}
              </p>
            </div>
            <div className="bg-dark-surface rounded-lg p-4">
              <p className="text-sm text-gray-400">Day Change</p>
              <p className={`text-2xl font-bold mt-1 ${activePortfolio.dayChange >= 0 ? 'text-success' : 'text-error'}`}>
                {activePortfolio.dayChange >= 0 ? '+' : ''}${Math.abs(activePortfolio.dayChange).toLocaleString()}
              </p>
            </div>
            <div className="bg-dark-surface rounded-lg p-4">
              <p className="text-sm text-gray-400">Total Return</p>
              <p className={`text-2xl font-bold mt-1 ${activePortfolio.totalProfitLoss >= 0 ? 'text-success' : 'text-error'}`}>
                {activePortfolio.totalProfitLoss >= 0 ? '+' : ''}${Math.abs(activePortfolio.totalProfitLoss).toLocaleString()}
              </p>
            </div>
            <div className="bg-dark-surface rounded-lg p-4">
              <p className="text-sm text-gray-400">Cash Balance</p>
              <p className="text-2xl font-bold text-gray-100 mt-1">
                ${activePortfolio.cash.toLocaleString()}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Positions Table */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-100 mb-4">Positions</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-dark-border">
                <th className="pb-3 text-sm font-medium text-gray-400">Symbol</th>
                <th className="pb-3 text-sm font-medium text-gray-400">Quantity</th>
                <th className="pb-3 text-sm font-medium text-gray-400">Avg Cost</th>
                <th className="pb-3 text-sm font-medium text-gray-400">Current Price</th>
                <th className="pb-3 text-sm font-medium text-gray-400">Market Value</th>
                <th className="pb-3 text-sm font-medium text-gray-400">P&L</th>
                <th className="pb-3 text-sm font-medium text-gray-400">P&L %</th>
              </tr>
            </thead>
            <tbody>
              {activePortfolio?.positions.map((position) => (
                <tr key={position.id} className="border-b border-dark-border">
                  <td className="py-4">
                    <div>
                      <p className="font-medium text-gray-100">{position.symbol}</p>
                      <p className="text-sm text-gray-400">{position.name}</p>
                    </div>
                  </td>
                  <td className="py-4 text-gray-300">{position.quantity}</td>
                  <td className="py-4 text-gray-300">${position.averagePrice.toFixed(2)}</td>
                  <td className="py-4 text-gray-300">${position.currentPrice.toFixed(2)}</td>
                  <td className="py-4 text-gray-300">${position.totalValue.toLocaleString()}</td>
                  <td className={`py-4 font-medium ${position.profitLoss >= 0 ? 'text-success' : 'text-error'}`}>
                    <div className="flex items-center">
                      {position.profitLoss >= 0 ? (
                        <TrendingUp className="w-4 h-4 mr-1" />
                      ) : (
                        <TrendingDown className="w-4 h-4 mr-1" />
                      )}
                      ${Math.abs(position.profitLoss).toLocaleString()}
                    </div>
                  </td>
                  <td className={`py-4 font-medium ${position.profitLossPercent >= 0 ? 'text-success' : 'text-error'}`}>
                    {position.profitLossPercent >= 0 ? '+' : ''}{position.profitLossPercent.toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default Portfolio