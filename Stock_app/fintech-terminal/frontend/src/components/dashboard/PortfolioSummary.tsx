import { useSelector } from 'react-redux'
import { RootState } from '@/store'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts'

const PortfolioSummary = () => {
  const { activePortfolio } = useSelector((state: RootState) => state.portfolio)

  // Mock allocation data
  const allocationData = [
    { name: 'Technology', value: 35, color: '#0ea5e9' },
    { name: 'Healthcare', value: 25, color: '#10b981' },
    { name: 'Finance', value: 20, color: '#f59e0b' },
    { name: 'Energy', value: 10, color: '#ef4444' },
    { name: 'Cash', value: 10, color: '#6b7280' },
  ]

  const topPositions = activePortfolio?.positions.slice(0, 5) || []

  return (
    <div className="card">
      <h2 className="text-xl font-semibold text-gray-100 mb-6">Portfolio Summary</h2>
      
      {/* Allocation Chart */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Asset Allocation</h3>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={allocationData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
            >
              {allocationData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(26, 31, 46, 0.95)',
                border: '1px solid #2a3441',
                borderRadius: '8px',
              }}
              labelStyle={{ color: '#e5e7eb' }}
            />
            <Legend
              verticalAlign="middle"
              align="right"
              layout="vertical"
              iconType="circle"
              formatter={(value, entry) => (
                <span className="text-gray-300 text-sm">
                  {value} ({entry.payload.value}%)
                </span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Top Holdings */}
      <div>
        <h3 className="text-sm font-medium text-gray-400 mb-4">Top Holdings</h3>
        <div className="space-y-3">
          {topPositions.map((position) => (
            <div key={position.id} className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-100">{position.symbol}</p>
                <p className="text-sm text-gray-400">{position.quantity} shares</p>
              </div>
              <div className="text-right">
                <p className="font-medium text-gray-100">
                  ${position.totalValue.toLocaleString()}
                </p>
                <p className={`text-sm ${position.profitLossPercent >= 0 ? 'text-success' : 'text-error'}`}>
                  {position.profitLossPercent >= 0 ? '+' : ''}{position.profitLossPercent.toFixed(2)}%
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default PortfolioSummary