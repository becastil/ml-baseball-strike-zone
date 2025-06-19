import api from './api'
import { Stock, MarketIndex, ChartData, NewsArticle, MarketSentiment, OrderBook } from '@/types'

export const marketService = {
  // Stock quotes - using Yahoo Finance via simplified backend
  async getQuote(symbol: string) {
    return api.get<Stock>(`/api/v1/market/quote/${symbol}`)
  },

  async getMultipleQuotes(symbols: string[]) {
    return api.get<{quotes: Stock[], count: number}>('/api/v1/market/quotes', { symbols: symbols.join(',') })
  },

  // Historical data
  async getHistoricalData(symbol: string, period: string = '1mo', interval: string = '1d') {
    return api.get<{symbol: string, data: ChartData[], count: number}>(`/api/v1/market/history/${symbol}`, { period, interval })
  },

  // Market indices
  async getMarketIndices() {
    return api.get<{indices: MarketIndex[]}>('/api/v1/market/indices')
  },

  // Symbol search
  async searchSymbols(query: string) {
    return api.get<{results: any[]}>('/api/v1/market/search', { query })
  },

  // WebSocket connection for real-time data
  connectToRealTimeData(onMessage: (data: any) => void) {
    const ws = new WebSocket('ws://localhost:8000/ws/market')
    
    ws.onopen = () => {
      console.log('Connected to real-time market data')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      onMessage(data)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    ws.onclose = () => {
      console.log('Disconnected from real-time market data')
    }

    return ws
  },

  // Real-time updates via polling (since Yahoo Finance doesn't provide WebSocket)
  startPolling(symbols: string[], onUpdate: (data: Stock[]) => void, intervalMs: number = 5000) {
    const poll = async () => {
      try {
        const response = await this.getMultipleQuotes(symbols)
        onUpdate(response.data)
      } catch (error) {
        console.error('Polling error:', error)
      }
    }
    
    // Initial fetch
    poll()
    
    // Set up interval
    const intervalId = setInterval(poll, intervalMs)
    
    // Return cleanup function
    return () => clearInterval(intervalId)
  },

  // Legacy methods for compatibility
  async getStocks(symbols: string[]) {
    return this.getMultipleQuotes(symbols)
  },

  async getStock(symbol: string) {
    return this.getQuote(symbol)
  },

  async searchStocks(query: string) {
    return this.searchSymbols(query)
  },

  async getChartData(symbol: string, interval: string, period?: string) {
    return this.getHistoricalData(symbol, period || '1mo', interval)
  },
}