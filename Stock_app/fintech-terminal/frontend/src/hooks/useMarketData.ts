import { useEffect, useRef, useCallback, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { RootState, AppDispatch } from '@/store'
import { updateStockPrice, setStocks } from '@/store/slices/marketSlice'

interface UseMarketDataOptions {
  symbols: string[]
  enabled?: boolean
  onUpdate?: (data: any) => void
  autoReconnect?: boolean
}

interface MarketUpdate {
  type: string
  data?: any[]
  symbols?: string[]
  message?: string
  timestamp: string
}

export const useMarketData = ({ 
  symbols, 
  enabled = true, 
  onUpdate,
  autoReconnect = true 
}: UseMarketDataOptions) => {
  const dispatch = useDispatch<AppDispatch>()
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const reconnectAttemptsRef = useRef(0)
  const subscribedSymbolsRef = useRef<Set<string>>(new Set())
  
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  const stocks = useSelector((state: RootState) => state.market.stocks)

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data: MarketUpdate = JSON.parse(event.data)
      console.log('📡 Received market update:', data.type, data)
      
      setLastUpdate(new Date())
      setError(null)
      
      switch (data.type) {
        case 'welcome':
          console.log('🎉 WebSocket connected:', data.message)
          break
          
        case 'initial_data':
        case 'market_update':
          if (data.data && Array.isArray(data.data)) {
            // Update Redux store with new market data
            dispatch(setStocks(data.data))
            
            // Update individual prices for smooth animations
            data.data.forEach((quote: any) => {
              dispatch(updateStockPrice({
                symbol: quote.symbol,
                price: quote.price,
                change: quote.change,
                changePercent: quote.changePercent
              }))
            })
          }
          break
          
        case 'subscribed':
          console.log('✅ Subscribed to:', data.symbols)
          if (data.symbols) {
            data.symbols.forEach(symbol => subscribedSymbolsRef.current.add(symbol))
          }
          break
          
        case 'unsubscribed':
          console.log('❌ Unsubscribed from:', data.symbols)
          if (data.symbols) {
            data.symbols.forEach(symbol => subscribedSymbolsRef.current.delete(symbol))
          }
          break
          
        case 'error':
          console.error('🚨 WebSocket error:', data.message)
          setError(data.message || 'Unknown error')
          break
          
        case 'pong':
          console.log('🏓 Pong received')
          break
      }
      
      // Call user-provided callback
      if (onUpdate) {
        onUpdate(data)
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error)
      setError('Failed to parse message')
    }
  }, [dispatch, onUpdate])

  const connect = useCallback(() => {
    if (!enabled) {
      console.log('🚫 Market data disabled')
      return
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('🔄 Already connected')
      return
    }

    try {
      console.log('🔌 Connecting to WebSocket...')
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000'
      wsRef.current = new WebSocket(`${wsUrl}/api/v1/realtime/ws`)

      wsRef.current.onopen = () => {
        console.log('🎉 WebSocket connected successfully')
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
        
        // Subscribe to symbols if any
        if (symbols.length > 0) {
          subscribe(symbols)
        }
      }

      wsRef.current.onmessage = handleMessage

      wsRef.current.onclose = (event) => {
        console.log('🔌 WebSocket closed:', event.code, event.reason)
        setIsConnected(false)
        
        // Implement exponential backoff for reconnection
        if (autoReconnect && reconnectAttemptsRef.current < 5) {
          const timeout = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000)
          console.log(`🔄 Reconnecting in ${timeout}ms...`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++
            connect()
          }, timeout)
        } else if (reconnectAttemptsRef.current >= 5) {
          setError('Max reconnection attempts reached')
        }
      }

      wsRef.current.onerror = (error) => {
        console.error('🚨 WebSocket error:', error)
        setError('WebSocket connection error')
        setIsConnected(false)
      }
    } catch (error) {
      console.error('🚨 Failed to create WebSocket:', error)
      setError('Failed to create WebSocket connection')
    }
  }, [enabled, symbols, handleMessage, autoReconnect])

  const disconnect = useCallback(() => {
    console.log('🔌 Disconnecting WebSocket...')
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'User initiated disconnect')
      wsRef.current = null
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    setIsConnected(false)
    subscribedSymbolsRef.current.clear()
  }, [])

  const subscribe = useCallback((newSymbols: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && newSymbols.length > 0) {
      console.log('📡 Subscribing to symbols:', newSymbols)
      wsRef.current.send(JSON.stringify({
        action: 'subscribe',
        symbols: newSymbols,
      }))
    } else if (newSymbols.length > 0) {
      console.log('⏳ WebSocket not ready, will subscribe when connected')
    }
  }, [])

  const unsubscribe = useCallback((symbolsToRemove: string[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && symbolsToRemove.length > 0) {
      console.log('📡 Unsubscribing from symbols:', symbolsToRemove)
      wsRef.current.send(JSON.stringify({
        action: 'unsubscribe',
        symbols: symbolsToRemove,
      }))
    }
  }, [])

  const ping = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'ping' }))
    }
  }, [])

  const getSubscriptions = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'get_subscriptions' }))
    }
  }, [])

  // Auto-connect on mount and symbol changes
  useEffect(() => {
    connect()
    
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  // Resubscribe when symbols change
  useEffect(() => {
    if (isConnected && symbols.length > 0) {
      const currentSymbols = Array.from(subscribedSymbolsRef.current)
      const newSymbols = symbols.filter(s => !currentSymbols.includes(s))
      const removedSymbols = currentSymbols.filter(s => !symbols.includes(s))
      
      if (removedSymbols.length > 0) {
        unsubscribe(removedSymbols)
      }
      
      if (newSymbols.length > 0) {
        subscribe(newSymbols)
      }
    }
  }, [symbols, isConnected, subscribe, unsubscribe])

  // Heartbeat ping every 30 seconds
  useEffect(() => {
    if (!isConnected) return
    
    const interval = setInterval(() => {
      ping()
    }, 30000)
    
    return () => clearInterval(interval)
  }, [isConnected, ping])

  return {
    // Data
    stocks,
    
    // Connection state
    isConnected,
    lastUpdate,
    error,
    subscribedSymbols: Array.from(subscribedSymbolsRef.current),
    
    // Actions
    subscribe,
    unsubscribe,
    connect,
    disconnect,
    ping,
    getSubscriptions,
    
    // Stats
    reconnectAttempts: reconnectAttemptsRef.current,
  }
}