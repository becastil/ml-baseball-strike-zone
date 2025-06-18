import api from './api'
import { Stock, MarketIndex, ChartData, NewsArticle, MarketSentiment, OrderBook } from '@/types'

export const marketService = {
  // Stock data
  async getStocks(symbols: string[]) {
    return api.get<Stock[]>('/stocks', { symbols: symbols.join(',') })
  },

  async getStock(symbol: string) {
    return api.get<Stock>(`/stocks/${symbol}`)
  },

  async searchStocks(query: string) {
    return api.get<Stock[]>('/stocks/search', { q: query })
  },

  // Market indices
  async getMarketIndices() {
    return api.get<MarketIndex[]>('/market/indices')
  },

  // Chart data
  async getChartData(symbol: string, interval: string, period?: string) {
    return api.get<ChartData[]>(`/stocks/${symbol}/chart`, { interval, period })
  },

  // Real-time data
  async getQuote(symbol: string) {
    return api.get<Stock>(`/stocks/${symbol}/quote`)
  },

  async getOrderBook(symbol: string) {
    return api.get<OrderBook>(`/stocks/${symbol}/orderbook`)
  },

  // News and sentiment
  async getMarketNews(limit: number = 20) {
    return api.get<NewsArticle[]>('/news/market', { limit })
  },

  async getStockNews(symbol: string, limit: number = 10) {
    return api.get<NewsArticle[]>(`/news/stocks/${symbol}`, { limit })
  },

  async getMarketSentiment() {
    return api.get<MarketSentiment>('/market/sentiment')
  },

  // Watchlist
  async getWatchlist() {
    return api.get<Stock[]>('/watchlist')
  },

  async addToWatchlist(symbol: string) {
    return api.post('/watchlist', { symbol })
  },

  async removeFromWatchlist(symbol: string) {
    return api.delete(`/watchlist/${symbol}`)
  },

  // WebSocket connections
  subscribeToMarketData(symbols: string[], onMessage: (data: any) => void) {
    const ws = api.createWebSocket('/market/stream')
    
    ws.onopen = () => {
      ws.send(JSON.stringify({ action: 'subscribe', symbols }))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      onMessage(data)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    return ws
  },

  // Historical data
  async getHistoricalData(symbol: string, startDate: string, endDate: string) {
    return api.get<ChartData[]>(`/stocks/${symbol}/historical`, { startDate, endDate })
  },

  // Market movers
  async getTopGainers(limit: number = 10) {
    return api.get<Stock[]>('/market/gainers', { limit })
  },

  async getTopLosers(limit: number = 10) {
    return api.get<Stock[]>('/market/losers', { limit })
  },

  async getMostActive(limit: number = 10) {
    return api.get<Stock[]>('/market/active', { limit })
  },
}