// Market Data Types
export interface Stock {
  symbol: string
  name: string
  price: number
  change: number
  changePercent: number
  volume: number
  marketCap: number
  high: number
  low: number
  open: number
  previousClose: number
  timestamp: string
}

export interface MarketIndex {
  symbol: string
  name: string
  value: number
  change: number
  changePercent: number
}

export interface ChartData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface WatchlistItem {
  id: string
  symbol: string
  name: string
  notes?: string
  targetPrice?: number
  alertEnabled: boolean
}

// Portfolio Types
export interface Position {
  id: string
  symbol: string
  name: string
  quantity: number
  averagePrice: number
  currentPrice: number
  totalValue: number
  profitLoss: number
  profitLossPercent: number
  dayChange: number
  dayChangePercent: number
}

export interface Transaction {
  id: string
  symbol: string
  type: 'BUY' | 'SELL'
  quantity: number
  price: number
  totalAmount: number
  timestamp: string
  fees: number
}

export interface Portfolio {
  id: string
  name: string
  totalValue: number
  dayChange: number
  dayChangePercent: number
  totalProfitLoss: number
  totalProfitLossPercent: number
  positions: Position[]
  cash: number
}

// API Response Types
export interface ApiResponse<T> {
  data: T
  status: number
  message?: string
}

export interface ApiError {
  message: string
  code: string
  details?: any
}

// User Types
export interface User {
  id: string
  email: string
  name: string
  avatarUrl?: string
  preferences: UserPreferences
}

export interface UserPreferences {
  theme: 'light' | 'dark'
  defaultView: 'dashboard' | 'market' | 'portfolio'
  notifications: {
    priceAlerts: boolean
    portfolioUpdates: boolean
    marketNews: boolean
  }
}

// News & Analytics Types
export interface NewsArticle {
  id: string
  title: string
  summary: string
  source: string
  url: string
  publishedAt: string
  sentiment?: 'positive' | 'negative' | 'neutral'
  relatedSymbols: string[]
}

export interface MarketSentiment {
  overall: 'bullish' | 'bearish' | 'neutral'
  score: number
  fearGreedIndex: number
  volatilityIndex: number
}

// Real-time Types
export interface PriceUpdate {
  symbol: string
  price: number
  volume: number
  timestamp: string
}

export interface OrderBook {
  symbol: string
  bids: OrderBookEntry[]
  asks: OrderBookEntry[]
  timestamp: string
}

export interface OrderBookEntry {
  price: number
  size: number
  total: number
}