import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const useAuthStore = create(
  persist(
    (set) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      role: null,          // ← top-level so ProtectedRoute can read it directly
      isAuthenticated: false,

      /**
       * Called after login or register.
       * user  — the user object from the API ({ id, email, role, firstName, lastName, ... })
       * accessToken / refreshToken — JWT strings
       */
      setAuth: (user, accessToken, refreshToken) =>
        set({
          user,
          accessToken,
          refreshToken,
          role: user?.role ?? null,
          isAuthenticated: !!(accessToken && user),
        }),

      setAccessToken: (accessToken) => set({ accessToken }),

      logout: () =>
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          role: null,
          isAuthenticated: false,
        }),
    }),
    { name: 'auth-storage' }
  )
)

export default useAuthStore