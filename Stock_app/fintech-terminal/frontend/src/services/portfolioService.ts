import api from './api'
import { Portfolio, Position, Transaction } from '@/types'

export const portfolioService = {
  // Portfolio management
  async getPortfolios() {
    return api.get<Portfolio[]>('/portfolios')
  },

  async getPortfolio(id: string) {
    return api.get<Portfolio>(`/portfolios/${id}`)
  },

  async createPortfolio(data: { name: string; initialCash: number }) {
    return api.post<Portfolio>('/portfolios', data)
  },

  async updatePortfolio(id: string, data: Partial<Portfolio>) {
    return api.put<Portfolio>(`/portfolios/${id}`, data)
  },

  async deletePortfolio(id: string) {
    return api.delete(`/portfolios/${id}`)
  },

  // Positions
  async getPositions(portfolioId: string) {
    return api.get<Position[]>(`/portfolios/${portfolioId}/positions`)
  },

  async getPosition(portfolioId: string, positionId: string) {
    return api.get<Position>(`/portfolios/${portfolioId}/positions/${positionId}`)
  },

  // Transactions
  async getTransactions(portfolioId: string, limit?: number, offset?: number) {
    return api.get<Transaction[]>(`/portfolios/${portfolioId}/transactions`, { limit, offset })
  },

  async addTransaction(transaction: Omit<Transaction, 'id' | 'timestamp'>) {
    return api.post<Transaction>('/transactions', transaction)
  },

  async getTransactionHistory(portfolioId: string, startDate?: string, endDate?: string) {
    return api.get<Transaction[]>(`/portfolios/${portfolioId}/transactions/history`, {
      startDate,
      endDate,
    })
  },

  // Performance analytics
  async getPortfolioPerformance(portfolioId: string, period: string = '1M') {
    return api.get(`/portfolios/${portfolioId}/performance`, { period })
  },

  async getPortfolioReturns(portfolioId: string, benchmark?: string) {
    return api.get(`/portfolios/${portfolioId}/returns`, { benchmark })
  },

  async getPortfolioAllocation(portfolioId: string) {
    return api.get(`/portfolios/${portfolioId}/allocation`)
  },

  // Risk metrics
  async getPortfolioRisk(portfolioId: string) {
    return api.get(`/portfolios/${portfolioId}/risk`)
  },

  // Export functionality
  async exportPortfolio(portfolioId: string, format: 'csv' | 'pdf' = 'csv') {
    return api.get(`/portfolios/${portfolioId}/export`, { format })
  },
}