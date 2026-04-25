import { Navigate } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'

export default function ProtectedRoute({ children, allowedRole }) {
<<<<<<< HEAD
  const { isAuthenticated, role } = useAuthStore()

  if (!isAuthenticated) return <Navigate to="/login" replace />
=======
  const { isAuthenticated, user } = useAuthStore()
  const role = user?.role ?? null

  if (!user) return <Navigate to="/login" replace />
>>>>>>> a259412 (frontend v2 not completed)
  if (allowedRole && role !== allowedRole) return <Navigate to="/login" replace />

  return children
}