import { useState } from 'react'
import { Search, Filter, Grid, List } from 'lucide-react'

const MarketView = () => {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')

  const categories = [
    { id: 'all', label: 'All Stocks' },
    { id: 'tech', label: 'Technology' },
    { id: 'finance', label: 'Finance' },
    { id: 'healthcare', label: 'Healthcare' },
    { id: 'energy', label: 'Energy' },
    { id: 'crypto', label: 'Crypto' },
  ]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Market View</h1>
          <p className="text-gray-400 mt-1">Explore stocks, indices, and market trends</p>
        </div>
        
        {/* View Toggle */}
        <div className="flex items-center space-x-2 bg-dark-surface rounded-lg p-1">
          <button
            onClick={() => setViewMode('grid')}
            className={`p-2 rounded ${viewMode === 'grid' ? 'bg-primary-600 text-white' : 'text-gray-400'}`}
          >
            <Grid className="w-5 h-5" />
          </button>
          <button
            onClick={() => setViewMode('list')}
            className={`p-2 rounded ${viewMode === 'list' ? 'bg-primary-600 text-white' : 'text-gray-400'}`}
          >
            <List className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="flex items-center space-x-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search stocks, symbols, or companies..."
            className="input pl-10 w-full"
          />
        </div>
        <button className="btn-secondary flex items-center space-x-2">
          <Filter className="w-5 h-5" />
          <span>Filters</span>
        </button>
      </div>

      {/* Category Tabs */}
      <div className="flex items-center space-x-2 border-b border-dark-border">
        {categories.map((category) => (
          <button
            key={category.id}
            onClick={() => setSelectedCategory(category.id)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              selectedCategory === category.id
                ? 'text-primary-400 border-primary-400'
                : 'text-gray-400 border-transparent hover:text-gray-200'
            }`}
          >
            {category.label}
          </button>
        ))}
      </div>

      {/* Market Content */}
      <div className="text-center py-12">
        <p className="text-gray-400">Market view content will be displayed here</p>
      </div>
    </div>
  )
}

export default MarketView