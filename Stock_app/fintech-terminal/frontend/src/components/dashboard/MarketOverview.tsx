import { useState } from 'react'
import { useSelector } from 'react-redux'
import { RootState } from '@/store'
import StockChart from '@/components/charts/StockChart'
import { ArrowUpRight, ArrowDownRight } from 'lucide-react'

const MarketOverview = () => {
  const { stocks, indices } = useSelector((state: RootState) => state.market)
  const [selectedTab, setSelectedTab] = useState<'gainers' | 'losers' | 'active'>('gainers')

  // Mock data for demonstration
  const marketMovers = {
    gainers: stocks.filter(s => s.changePercent > 0).sort((a, b) => b.changePercent - a.changePercent).slice(0, 5),
    losers: stocks.filter(s => s.changePercent < 0).sort((a, b) => a.changePercent - b.changePercent).slice(0, 5),
    active: stocks.sort((a, b) => b.volume - a.volume).slice(0, 5),
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-100">Market Overview</h2>
        <div className="flex space-x-2">
          <button
            onClick={() => setSelectedTab('gainers')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              selectedTab === 'gainers'
                ? 'bg-success/20 text-success'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Top Gainers
          </button>
          <button
            onClick={() => setSelectedTab('losers')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              selectedTab === 'losers'
                ? 'bg-error/20 text-error'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Top Losers
          </button>
          <button
            onClick={() => setSelectedTab('active')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              selectedTab === 'active'
                ? 'bg-primary-600/20 text-primary-400'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            Most Active
          </button>
        </div>
      </div>

      {/* Chart Section */}
      <div className="mb-6">
        <StockChart symbol="SPY" height={300} />
      </div>

      {/* Market Movers List */}
      <div className="space-y-3">
        {marketMovers[selectedTab].map((stock) => (
          <div
            key={stock.symbol}
            className="flex items-center justify-between p-3 bg-dark-surface rounded-lg hover:bg-dark-border transition-colors cursor-pointer"
          >
            <div className="flex items-center space-x-3">
              <div>
                <p className="font-medium text-gray-100">{stock.symbol}</p>
                <p className="text-sm text-gray-400">{stock.name}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <p className="font-medium text-gray-100">${stock.price.toFixed(2)}</p>
                <p className="text-sm text-gray-400">
                  Vol: {(stock.volume / 1000000).toFixed(2)}M
                </p>
              </div>
              
              <div className={`flex items-center ${stock.changePercent >= 0 ? 'text-success' : 'text-error'}`}>
                {stock.changePercent >= 0 ? (
                  <ArrowUpRight className="w-4 h-4 mr-1" />
                ) : (
                  <ArrowDownRight className="w-4 h-4 mr-1" />
                )}
                <span className="font-medium">
                  {Math.abs(stock.changePercent).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default MarketOverview