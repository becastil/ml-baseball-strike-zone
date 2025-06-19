import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { Stock, MarketIndex, WatchlistItem, ChartData } from '@/types'
import { marketService } from '@/services/marketService'

interface MarketState {
  stocks: Stock[]
  indices: MarketIndex[]
  watchlist: WatchlistItem[]
  selectedStock: Stock | null
  chartData: ChartData[]
  loading: boolean
  error: string | null
}

const initialState: MarketState = {
  stocks: [],
  indices: [],
  watchlist: [],
  selectedStock: null,
  chartData: [],
  loading: false,
  error: null,
}

// Async thunks
export const fetchStocks = createAsyncThunk(
  'market/fetchStocks',
  async (symbols: string[]) => {
    const response = await marketService.getStocks(symbols)
    return response.data
  }
)

export const fetchMarketIndices = createAsyncThunk(
  'market/fetchIndices',
  async () => {
    const response = await marketService.getMarketIndices()
    return response.data
  }
)

export const fetchChartData = createAsyncThunk(
  'market/fetchChartData',
  async ({ symbol, interval }: { symbol: string; interval: string }) => {
    const response = await marketService.getChartData(symbol, interval)
    return response.data
  }
)

const marketSlice = createSlice({
  name: 'market',
  initialState,
  reducers: {
    setSelectedStock: (state, action: PayloadAction<Stock>) => {
      state.selectedStock = action.payload
    },
    addToWatchlist: (state, action: PayloadAction<WatchlistItem>) => {
      state.watchlist.push(action.payload)
    },
    removeFromWatchlist: (state, action: PayloadAction<string>) => {
      state.watchlist = state.watchlist.filter(item => item.id !== action.payload)
    },
    updateStockPrice: (state, action: PayloadAction<{ symbol: string; price: number; change: number; changePercent?: number }>) => {
      const stock = state.stocks.find(s => s.symbol === action.payload.symbol)
      if (stock) {
        stock.price = action.payload.price
        stock.change = action.payload.change
        stock.changePercent = action.payload.changePercent || (action.payload.change / stock.previousClose) * 100
      }
    },
    setStocks: (state, action: PayloadAction<Stock[]>) => {
      state.stocks = action.payload
    },
    updateMultipleStocks: (state, action: PayloadAction<Stock[]>) => {
      action.payload.forEach(updatedStock => {
        const existingIndex = state.stocks.findIndex(s => s.symbol === updatedStock.symbol)
        if (existingIndex >= 0) {
          state.stocks[existingIndex] = { ...state.stocks[existingIndex], ...updatedStock }
        } else {
          state.stocks.push(updatedStock)
        }
      })
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch stocks
      .addCase(fetchStocks.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchStocks.fulfilled, (state, action) => {
        state.loading = false
        state.stocks = action.payload
      })
      .addCase(fetchStocks.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || 'Failed to fetch stocks'
      })
      // Fetch indices
      .addCase(fetchMarketIndices.fulfilled, (state, action) => {
        state.indices = action.payload
      })
      // Fetch chart data
      .addCase(fetchChartData.fulfilled, (state, action) => {
        state.chartData = action.payload
      })
  },
})

export const { setSelectedStock, addToWatchlist, removeFromWatchlist, updateStockPrice, setStocks, updateMultipleStocks } = marketSlice.actions
export default marketSlice.reducer