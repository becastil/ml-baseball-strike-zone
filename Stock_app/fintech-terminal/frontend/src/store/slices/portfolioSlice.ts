import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { Portfolio, Position, Transaction } from '@/types'
import { portfolioService } from '@/services/portfolioService'

interface PortfolioState {
  portfolios: Portfolio[]
  activePortfolio: Portfolio | null
  transactions: Transaction[]
  loading: boolean
  error: string | null
}

const initialState: PortfolioState = {
  portfolios: [],
  activePortfolio: null,
  transactions: [],
  loading: false,
  error: null,
}

// Async thunks
export const fetchPortfolios = createAsyncThunk(
  'portfolio/fetchPortfolios',
  async () => {
    const response = await portfolioService.getPortfolios()
    return response.data
  }
)

export const fetchTransactions = createAsyncThunk(
  'portfolio/fetchTransactions',
  async (portfolioId: string) => {
    const response = await portfolioService.getTransactions(portfolioId)
    return response.data
  }
)

export const addTransaction = createAsyncThunk(
  'portfolio/addTransaction',
  async (transaction: Omit<Transaction, 'id' | 'timestamp'>) => {
    const response = await portfolioService.addTransaction(transaction)
    return response.data
  }
)

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    setActivePortfolio: (state, action: PayloadAction<Portfolio>) => {
      state.activePortfolio = action.payload
    },
    updatePosition: (state, action: PayloadAction<{ portfolioId: string; position: Position }>) => {
      const portfolio = state.portfolios.find(p => p.id === action.payload.portfolioId)
      if (portfolio) {
        const positionIndex = portfolio.positions.findIndex(p => p.id === action.payload.position.id)
        if (positionIndex !== -1) {
          portfolio.positions[positionIndex] = action.payload.position
        }
      }
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch portfolios
      .addCase(fetchPortfolios.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchPortfolios.fulfilled, (state, action) => {
        state.loading = false
        state.portfolios = action.payload
        if (action.payload.length > 0 && !state.activePortfolio) {
          state.activePortfolio = action.payload[0]
        }
      })
      .addCase(fetchPortfolios.rejected, (state, action) => {
        state.loading = false
        state.error = action.error.message || 'Failed to fetch portfolios'
      })
      // Fetch transactions
      .addCase(fetchTransactions.fulfilled, (state, action) => {
        state.transactions = action.payload
      })
      // Add transaction
      .addCase(addTransaction.fulfilled, (state, action) => {
        state.transactions.unshift(action.payload)
      })
  },
})

export const { setActivePortfolio, updatePosition } = portfolioSlice.actions
export default portfolioSlice.reducer