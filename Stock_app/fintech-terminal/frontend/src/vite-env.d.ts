/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_WS_URL: string
  readonly VITE_ENABLE_MOCK_DATA: string
  readonly VITE_ENABLE_WEBSOCKET: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}