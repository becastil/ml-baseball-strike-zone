import React, { useState, useEffect } from 'react'
import { useMarketData } from '@/hooks/useMarketData'
import { Activity, TrendingUp, TrendingDown, Wifi, WifiOff, RefreshCw } from 'lucide-react'

interface Stock {
  symbol: string
  price: number
  change: number
  changePercent: number
  timestamp: string
}

const RealTimeMarketWidget: React.FC = () => {
  const [watchlist] = useState(['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META'])
  const [updates, setUpdates] = useState<string[]>([])
  
  const {
    stocks,
    isConnected,
    lastUpdate,
    error,
    subscribedSymbols,
    subscribe,
    connect,
    reconnectAttempts
  } = useMarketData({
    symbols: watchlist,
    enabled: true,
    onUpdate: (data) => {
      // Add update to activity log
      const updateText = `${new Date().toLocaleTimeString()}: ${data.type} - ${
        data.symbols ? data.symbols.join(', ') : 'market data'
      }`
      setUpdates(prev => [updateText, ...prev.slice(0, 9)]) // Keep last 10 updates
    }
  })

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(price)
  }

  const formatChange = (change: number, changePercent: number) => {
    const sign = change >= 0 ? '+' : ''
    return `${sign}${change.toFixed(2)} (${sign}${changePercent.toFixed(2)}%)`
  }

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-green-400'
    if (change < 0) return 'text-red-400'
    return 'text-gray-400'
  }

  const getChangeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="w-4 h-4" />
    if (change < 0) return <TrendingDown className="w-4 h-4" />
    return null
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 text-white">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-2">
          <Activity className="w-6 h-6" />
          Real-Time Market Data
        </h2>
        
        {/* Connection Status */}
        <div className="flex items-center gap-2">
          {isConnected ? (
            <div className="flex items-center gap-1 text-green-400">
              <Wifi className="w-4 h-4" />
              <span className="text-sm">Live</span>
            </div>
          ) : (
            <div className="flex items-center gap-1 text-red-400">
              <WifiOff className="w-4 h-4" />
              <span className="text-sm">
                {error ? 'Error' : `Reconnecting (${reconnectAttempts})`}
              </span>
            </div>
          )}
          
          <button
            onClick={connect}
            className="p-1 text-gray-400 hover:text-white transition-colors"
            title="Reconnect"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/20 border border-red-500 rounded-lg p-3 mb-4">
          <p className="text-red-200 text-sm">{error}</p>
        </div>
      )}

      {/* Last Update */}
      {lastUpdate && (
        <div className="text-sm text-gray-400 mb-4">
          Last update: {lastUpdate.toLocaleTimeString()}
        </div>
      )}

      {/* Stock Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {stocks.map((stock: Stock) => (
          <div
            key={stock.symbol}
            className="bg-gray-700 rounded-lg p-4 border border-gray-600 hover:border-gray-500 transition-colors"
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-bold text-lg">{stock.symbol}</h3>
              <div className={`flex items-center gap-1 ${getChangeColor(stock.change)}`}>
                {getChangeIcon(stock.change)}
              </div>
            </div>
            
            <div className="text-2xl font-bold mb-1">
              {formatPrice(stock.price)}
            </div>
            
            <div className={`text-sm ${getChangeColor(stock.change)}`}>
              {formatChange(stock.change, stock.changePercent)}
            </div>
            
            {stock.timestamp && (
              <div className="text-xs text-gray-500 mt-2">
                {new Date(stock.timestamp).toLocaleTimeString()}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Subscribed Symbols */}
      {subscribedSymbols.length > 0 && (
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-300 mb-2">
            Subscribed Symbols ({subscribedSymbols.length})
          </h4>
          <div className="flex flex-wrap gap-2">
            {subscribedSymbols.map(symbol => (
              <span
                key={symbol}
                className="px-2 py-1 bg-blue-600 rounded text-xs"
              >
                {symbol}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Activity Log */}
      <div>
        <h4 className="text-sm font-semibold text-gray-300 mb-2">
          Recent Activity
        </h4>
        <div className="bg-gray-900 rounded-lg p-3 max-h-32 overflow-y-auto">
          {updates.length > 0 ? (
            <div className="space-y-1">
              {updates.map((update, index) => (
                <div key={index} className="text-xs text-gray-400">
                  {update}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-gray-500">
              No updates yet. Waiting for market data...
            </div>
          )}
        </div>
      </div>

      {/* Instructions */}
      <div className="mt-4 p-3 bg-blue-600/20 border border-blue-600 rounded-lg">
        <h4 className="text-sm font-semibold text-blue-200 mb-1">
          📡 Real-Time Updates Every 5 Seconds
        </h4>
        <p className="text-xs text-blue-300">
          This widget connects to your backend via WebSocket for live market data. 
          Open your browser's developer console to see detailed connection logs.
        </p>
      </div>
    </div>
  )
}

export default RealTimeMarketWidget