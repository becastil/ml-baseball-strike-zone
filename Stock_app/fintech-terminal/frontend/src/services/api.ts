import axios, { AxiosInstance, AxiosError, AxiosResponse } from 'axios'
import { ApiResponse, ApiError } from '@/types'

class ApiService {
  private axiosInstance: AxiosInstance

  constructor() {
    this.axiosInstance = axios.create({
      baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Request interceptor
    this.axiosInstance.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('authToken')
        if (token) {
          config.headers.Authorization = `Bearer ${token}`
        }
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    // Response interceptor
    this.axiosInstance.interceptors.response.use(
      (response: AxiosResponse) => {
        return response
      },
      (error: AxiosError<ApiError>) => {
        if (error.response) {
          // Server responded with error
          const apiError: ApiError = {
            message: error.response.data?.message || 'An error occurred',
            code: error.response.data?.code || 'UNKNOWN_ERROR',
            details: error.response.data?.details,
          }
          
          // Handle specific error cases
          if (error.response.status === 401) {
            // Unauthorized - redirect to login
            localStorage.removeItem('authToken')
            window.location.href = '/login'
          }
          
          return Promise.reject(apiError)
        } else if (error.request) {
          // Request made but no response
          const apiError: ApiError = {
            message: 'Network error - please check your connection',
            code: 'NETWORK_ERROR',
          }
          return Promise.reject(apiError)
        } else {
          // Something else happened
          const apiError: ApiError = {
            message: error.message || 'An unexpected error occurred',
            code: 'CLIENT_ERROR',
          }
          return Promise.reject(apiError)
        }
      }
    )
  }

  // Generic request methods
  async get<T>(url: string, params?: any): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.get<T>(url, { params })
    return {
      data: response.data,
      status: response.status,
    }
  }

  async post<T>(url: string, data?: any): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.post<T>(url, data)
    return {
      data: response.data,
      status: response.status,
    }
  }

  async put<T>(url: string, data?: any): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.put<T>(url, data)
    return {
      data: response.data,
      status: response.status,
    }
  }

  async delete<T>(url: string): Promise<ApiResponse<T>> {
    const response = await this.axiosInstance.delete<T>(url)
    return {
      data: response.data,
      status: response.status,
    }
  }

  // WebSocket connection helper
  createWebSocket(path: string): WebSocket {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'
    const token = localStorage.getItem('authToken')
    const url = `${wsUrl}${path}${token ? `?token=${token}` : ''}`
    return new WebSocket(url)
  }
}

export const api = new ApiService()
export default api