import { NavLink } from 'react-router-dom'
import { 
  LayoutDashboard, 
  TrendingUp, 
  Briefcase, 
  BarChart3, 
  Newspaper, 
  Settings,
  HelpCircle,
  LogOut
} from 'lucide-react'

const Sidebar = () => {
  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/market', icon: TrendingUp, label: 'Market' },
    { path: '/portfolio', icon: Briefcase, label: 'Portfolio' },
    { path: '/analytics', icon: BarChart3, label: 'Analytics' },
    { path: '/news', icon: Newspaper, label: 'News' },
  ]

  const bottomItems = [
    { path: '/settings', icon: Settings, label: 'Settings' },
    { path: '/help', icon: HelpCircle, label: 'Help' },
  ]

  return (
    <div className="w-64 bg-dark-surface border-r border-dark-border flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-dark-border">
        <h1 className="text-2xl font-bold text-gradient">FinTech Terminal</h1>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-primary-600/20 text-primary-400 border-l-4 border-primary-500'
                      : 'text-gray-400 hover:bg-dark-border hover:text-gray-200'
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        {/* Watchlist */}
        <div className="mt-8">
          <h3 className="px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            Watchlist
          </h3>
          <ul className="space-y-1">
            <li className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 cursor-pointer flex justify-between items-center">
              <span>AAPL</span>
              <span className="text-success text-xs">+2.34%</span>
            </li>
            <li className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 cursor-pointer flex justify-between items-center">
              <span>GOOGL</span>
              <span className="text-error text-xs">-1.23%</span>
            </li>
            <li className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 cursor-pointer flex justify-between items-center">
              <span>MSFT</span>
              <span className="text-success text-xs">+0.56%</span>
            </li>
          </ul>
        </div>
      </nav>

      {/* Bottom Section */}
      <div className="px-4 py-6 border-t border-dark-border">
        <ul className="space-y-2">
          {bottomItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    isActive
                      ? 'bg-primary-600/20 text-primary-400'
                      : 'text-gray-400 hover:bg-dark-border hover:text-gray-200'
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </NavLink>
            </li>
          ))}
          <li>
            <button className="flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 text-gray-400 hover:bg-dark-border hover:text-gray-200 w-full">
              <LogOut className="w-5 h-5" />
              <span className="font-medium">Logout</span>
            </button>
          </li>
        </ul>
      </div>
    </div>
  )
}

export default Sidebar