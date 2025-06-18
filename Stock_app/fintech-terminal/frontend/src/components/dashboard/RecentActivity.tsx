import { useSelector } from 'react-redux'
import { RootState } from '@/store'
import { Clock, TrendingUp, TrendingDown, DollarSign } from 'lucide-react'
import { format } from 'date-fns'

const RecentActivity = () => {
  const { transactions } = useSelector((state: RootState) => state.portfolio)

  // Mock activities for demonstration
  const activities = [
    {
      id: '1',
      type: 'BUY',
      symbol: 'AAPL',
      quantity: 10,
      price: 150.25,
      timestamp: new Date().toISOString(),
      icon: TrendingUp,
      color: 'text-success',
      bgColor: 'bg-success/20',
    },
    {
      id: '2',
      type: 'SELL',
      symbol: 'GOOGL',
      quantity: 5,
      price: 2850.50,
      timestamp: new Date(Date.now() - 86400000).toISOString(),
      icon: TrendingDown,
      color: 'text-error',
      bgColor: 'bg-error/20',
    },
    {
      id: '3',
      type: 'DIVIDEND',
      symbol: 'MSFT',
      amount: 125.00,
      timestamp: new Date(Date.now() - 172800000).toISOString(),
      icon: DollarSign,
      color: 'text-primary-400',
      bgColor: 'bg-primary-600/20',
    },
  ]

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-100">Recent Activity</h2>
        <button className="text-sm text-primary-400 hover:text-primary-300">
          View All
        </button>
      </div>

      <div className="space-y-4">
        {activities.map((activity) => (
          <div key={activity.id} className="flex items-start space-x-3">
            <div className={`p-2 rounded-lg ${activity.bgColor}`}>
              <activity.icon className={`w-5 h-5 ${activity.color}`} />
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-100">
                  {activity.type === 'DIVIDEND' ? (
                    <>Dividend from {activity.symbol}</>
                  ) : (
                    <>
                      {activity.type} {activity.quantity} {activity.symbol}
                      {activity.price && ` @ $${activity.price.toFixed(2)}`}
                    </>
                  )}
                </p>
                <p className="text-sm font-medium text-gray-100">
                  {activity.type === 'DIVIDEND' ? (
                    `+$${activity.amount.toFixed(2)}`
                  ) : (
                    `$${(activity.quantity * activity.price).toFixed(2)}`
                  )}
                </p>
              </div>
              
              <div className="flex items-center mt-1">
                <Clock className="w-3 h-3 text-gray-500 mr-1" />
                <p className="text-xs text-gray-500">
                  {format(new Date(activity.timestamp), 'MMM d, yyyy h:mm a')}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default RecentActivity