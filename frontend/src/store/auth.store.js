import { create } from 'zustand'
import { persist } from 'zustand/middleware'

<<<<<<< HEAD
const useAuthStore = create(
  persist(
    (set) => ({
=======
/**
 * Auth store
 *
 * FIX 1 — added `role` getter derived from `user.role`
 *          so ProtectedRoute and any component can read it directly.
 *
 * FIX 2 — `setAuth` now also accepts a plain accessToken string as second
 *          arg (how AuthPage calls it) AND keeps backward compat with the
 *          ProfilePage pattern that passes (updatedUser, token, _ignoredRole).
 *          Role is ALWAYS derived from user.role, never stored separately.
 */
const useAuthStore = create(
  persist(
    (set, get) => ({
>>>>>>> a259412 (frontend v2 not completed)
      user: null,
      accessToken: null,
      refreshToken: null,

<<<<<<< HEAD
      setAuth: (user, accessToken, refreshToken) =>
        set({ user, accessToken, refreshToken }),
=======
      /** Derived — always reflects user.role so no separate field can drift */
      get role() {
        return get().user?.role ?? null
      },

      /** Called on login/register and after profile update.
       *  Signature: setAuth(user, accessToken, refreshToken?)
       *  The third argument is optional — ProfilePage omits it on profile update. */
      setAuth: (user, accessToken, refreshToken) =>
        set((state) => ({
          user,
          accessToken,
          // Keep existing refreshToken when caller doesn't supply one
          refreshToken: refreshToken !== undefined ? refreshToken : state.refreshToken,
        })),
>>>>>>> a259412 (frontend v2 not completed)

      setAccessToken: (accessToken) => set({ accessToken }),

      logout: () => set({ user: null, accessToken: null, refreshToken: null }),
    }),
<<<<<<< HEAD
    { name: 'auth-storage' }
=======
    {
      name: 'auth-storage',
      // Only persist primitives — role is derived on rehydration
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
      }),
    }
>>>>>>> a259412 (frontend v2 not completed)
  )
)

export default useAuthStore