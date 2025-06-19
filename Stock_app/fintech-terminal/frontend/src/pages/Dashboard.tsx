import { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { RootState, AppDispatch } from '@/store'
import { fetchStocks, fetchMarketIndices } from '@/store/slices/marketSlice'
import { fetchPortfolios } from '@/store/slices/portfolioSlice'
import MarketOverview from '@/components/dashboard/MarketOverview'
import PortfolioSummary from '@/components/dashboard/PortfolioSummary'
import RecentActivity from '@/components/dashboard/RecentActivity'
import QuickActions from '@/components/dashboard/QuickActions'
import RealTimeMarketWidget from '@/components/dashboard/RealTimeMarketWidget'
import { TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react'

const Dashboard = () => {
  const dispatch = useDispatch<AppDispatch>()
  const { stocks, indices, loading } = useSelector((state: RootState) => state.market)
  const { activePortfolio } = useSelector((state: RootState) => state.portfolio)

  useEffect(() => {
    // Fetch initial data
    dispatch(fetchMarketIndices())
    dispatch(fetchStocks(['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']))
    dispatch(fetchPortfolios())
  }, [dispatch])

  const stats = [
    {
      title: 'Portfolio Value',
      value: activePortfolio ? `$${activePortfolio.totalValue.toLocaleString()}` : '$0',
      change: activePortfolio?.dayChangePercent || 0,
      icon: DollarSign,
    },
    {
      title: 'Day Gain/Loss',
      value: activePortfolio ? `$${activePortfolio.dayChange.toLocaleString()}` : '$0',
      change: activePortfolio?.dayChangePercent || 0,
      icon: activePortfolio && activePortfolio.dayChange >= 0 ? TrendingUp : TrendingDown,
    },
    {
      title: 'Total Return',
      value: activePortfolio ? `$${activePortfolio.totalProfitLoss.toLocaleString()}` : '$0',
      change: activePortfolio?.totalProfitLossPercent || 0,
      icon: Activity,
    },
    {
      title: 'Cash Balance',
      value: activePortfolio ? `$${activePortfolio.cash.toLocaleString()}` : '$0',
      change: 0,
      icon: DollarSign,
    },
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-100">Dashboard</h1>
        <p className="text-gray-400 mt-1">Welcome back! Here's your market overview.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <div key={index} className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">{stat.title}</p>
                <p className="text-2xl font-bold text-gray-100 mt-1">{stat.value}</p>
                <p className={`text-sm mt-2 ${stat.change >= 0 ? 'text-success' : 'text-error'}`}>
                  {stat.change >= 0 ? '+' : ''}{stat.change.toFixed(2)}%
                </p>
              </div>
              <div className={`p-3 rounded-lg ${stat.change >= 0 ? 'bg-success/20' : 'bg-error/20'}`}>
                <stat.icon className={`w-6 h-6 ${stat.change >= 0 ? 'text-success' : 'text-error'}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Market Overview - Takes 2 columns */}
        <div className="lg:col-span-2">
          <MarketOverview />
        </div>
        
        {/* Quick Actions - Takes 1 column */}
        <div>
          <QuickActions />
        </div>
      </div>

      {/* Real-Time Market Data Widget */}
      <div>
        <RealTimeMarketWidget />
      </div>

      {/* Bottom Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Summary */}
        <PortfolioSummary />
        
        {/* Recent Activity */}
        <RecentActivity />
      </div>
    </div>
  )
}

export default Dashboard