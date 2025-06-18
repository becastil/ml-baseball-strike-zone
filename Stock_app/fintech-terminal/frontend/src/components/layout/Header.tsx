import { useState } from 'react'
import { useSelector } from 'react-redux'
import { RootState } from '@/store'
import { Bell, Search, Settings, User } from 'lucide-react'

const Header = () => {
  const [searchQuery, setSearchQuery] = useState('')
  const user = useSelector((state: RootState) => state.user.user)
  const indices = useSelector((state: RootState) => state.market.indices)

  return (
    <header className="bg-dark-surface border-b border-dark-border">
      <div className="flex items-center justify-between px-6 py-4">
        {/* Market Indices */}
        <div className="flex items-center space-x-6">
          {indices.map((index) => (
            <div key={index.symbol} className="flex items-center space-x-2">
              <span className="text-sm text-gray-400">{index.symbol}</span>
              <span className="text-sm font-medium">{index.value.toFixed(2)}</span>
              <span
                className={`text-sm font-medium ${
                  index.change >= 0 ? 'text-success' : 'text-error'
                }`}
              >
                {index.change >= 0 ? '+' : ''}{index.changePercent.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>

        {/* Search Bar */}
        <div className="flex-1 max-w-md mx-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search stocks, symbols..."
              className="input pl-10 w-full"
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center space-x-4">
          {/* Notifications */}
          <button className="relative p-2 text-gray-400 hover:text-gray-200 transition-colors">
            <Bell className="w-5 h-5" />
            <span className="absolute top-0 right-0 w-2 h-2 bg-primary-500 rounded-full"></span>
          </button>

          {/* Settings */}
          <button className="p-2 text-gray-400 hover:text-gray-200 transition-colors">
            <Settings className="w-5 h-5" />
          </button>

          {/* User Profile */}
          <div className="flex items-center space-x-3 pl-4 border-l border-dark-border">
            <div className="text-right">
              <p className="text-sm font-medium text-gray-200">{user?.name || 'Guest'}</p>
              <p className="text-xs text-gray-400">{user?.email || 'guest@example.com'}</p>
            </div>
            <div className="w-10 h-10 bg-primary-600 rounded-full flex items-center justify-center">
              {user?.avatarUrl ? (
                <img src={user.avatarUrl} alt={user.name} className="w-full h-full rounded-full" />
              ) : (
                <User className="w-5 h-5 text-white" />
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header