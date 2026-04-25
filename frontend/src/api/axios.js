import axios from 'axios'
import useAuthStore from '../store/auth.store'

const api = axios.create({
  baseURL: 'http://localhost:3000/api/v1',
})

// Attach JWT to every outgoing request
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().accessToken
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// On 401: try to refresh once, then logout
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config

    if (error.response?.status === 401 && !original._retry) {
      original._retry = true
      try {
        const refreshToken = useAuthStore.getState().refreshToken
        if (!refreshToken) throw new Error('No refresh token')

        const res = await axios.post('http://localhost:3000/api/v1/auth/refresh', {
          refreshToken,
        })

        // Backend wraps in { success, data: { accessToken, refreshToken, ... } }
        const payload = res.data?.data ?? res.data
        const newAccessToken = payload.accessToken

        useAuthStore.getState().setAccessToken(newAccessToken)
        original.headers.Authorization = `Bearer ${newAccessToken}`
        return api(original)
      } catch {
        // Refresh failed — full logout
        useAuthStore.getState().logout()
        window.location.href = '/login'
      }
    }

    return Promise.reject(error)
  }
)

export default api