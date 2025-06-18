import { useEffect, useRef, useCallback } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { RootState, AppDispatch } from '@/store'
import { updateStockPrice } from '@/store/slices/marketSlice'
import { marketService } from '@/services/marketService'

interface UseMarketDataOptions {
  symbols: string[]
  enabled?: boolean
  onUpdate?: (data: any) => void
}

export const useMarketData = ({ symbols, enabled = true, onUpdate }: UseMarketDataOptions) => {
  const dispatch = useDispatch<AppDispatch>()
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const reconnectAttemptsRef = useRef(0)

  const stocks = useSelector((state: RootState) => state.market.stocks)

  const handleMessage = useCallback((data: any) => {
    if (data.type === 'price_update') {
      dispatch(updateStockPrice({
        symbol: data.symbol,
        price: data.price,
        change: data.change,
      }))
    }
    
    if (onUpdate) {
      onUpdate(data)
    }
  }, [dispatch, onUpdate])

  const connect = useCallback(() => {
    if (!enabled || symbols.length === 0) return

    try {
      wsRef.current = marketService.subscribeToMarketData(symbols, handleMessage)

      wsRef.current.onclose = () => {
        console.log('WebSocket closed')
        
        // Implement exponential backoff for reconnection
        if (reconnectAttemptsRef.current < 5) {
          const timeout = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000)
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttemptsRef.current++
            connect()
          }, timeout)
        }
      }

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        reconnectAttemptsRef.current = 0
      }
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error)
    }
  }, [enabled, symbols, handleMessage])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
  }, [])

  useEffect(() => {
    connect()
    
    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  // Subscribe/unsubscribe to symbols
  const subscribe = useCallback((newSymbols: string[]) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'subscribe',
        symbols: newSymbols,
      }))
    }
  }, [])

  const unsubscribe = useCallback((symbolsToRemove: string[]) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'unsubscribe',
        symbols: symbolsToRemove,
      }))
    }
  }, [])

  return {
    stocks,
    subscribe,
    unsubscribe,
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  }
}