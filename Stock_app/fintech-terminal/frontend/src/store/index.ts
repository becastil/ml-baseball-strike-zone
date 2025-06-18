import { configureStore } from '@reduxjs/toolkit'
import marketReducer from './slices/marketSlice'
import portfolioReducer from './slices/portfolioSlice'
import userReducer from './slices/userSlice'

export const store = configureStore({
  reducer: {
    market: marketReducer,
    portfolio: portfolioReducer,
    user: userReducer,
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch