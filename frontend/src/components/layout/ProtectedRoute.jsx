import { Navigate } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'

export default function ProtectedRoute({ children, allowedRole }) {
  const { isAuthenticated, role } = useAuthStore()

  if (!isAuthenticated) return <Navigate to="/login" replace />
  if (allowedRole && role !== allowedRole) return <Navigate to="/login" replace />

  return children
}