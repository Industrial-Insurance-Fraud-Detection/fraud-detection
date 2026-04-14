import axios from 'axios'
import useAuthStore from '../store/auth.store'

const api = axios.create({
  baseURL: 'http://localhost:3000/api/v1',
})

// ✅ Attache automatiquement le token JWT à chaque requête
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().accessToken
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// ✅ Si le token expire (401), tente un refresh automatique
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config
    if (error.response?.status === 401 && !original._retry) {
      original._retry = true
      try {
        const refreshToken = useAuthStore.getState().refreshToken
        const res = await axios.post('http://localhost:3000/api/v1/auth/refresh', {
          refreshToken,
        })
        const newToken = res.data?.data?.accessToken || res.data?.accessToken
        useAuthStore.getState().setAccessToken(newToken)
        original.headers.Authorization = `Bearer ${newToken}`
        return api(original)
      } catch {
        useAuthStore.getState().logout()
        window.location.href = '/'
      }
    }
    return Promise.reject(error)
  }
)

export default api