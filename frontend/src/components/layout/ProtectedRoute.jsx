import { Navigate } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'

/**
 * Guards a route by authentication state and optional role.
 *
 * Usage:
 *   <ProtectedRoute allowedRole="CLIENT">  <ClientDashboard /> </ProtectedRoute>
 *   <ProtectedRoute allowedRole="INVESTIGATOR"> <InvDashboard /> </ProtectedRoute>
 */
export default function ProtectedRoute({ children, allowedRole }) {
  const { isAuthenticated, role } = useAuthStore()

  // Not logged in → back to login
  if (!isAuthenticated) return <Navigate to="/login" replace />

  // Wrong role (e.g. client trying to reach /investigator/*)
  if (allowedRole && role !== allowedRole) {
    // Send them to their own dashboard instead of back to login
    if (role === 'CLIENT') return <Navigate to="/client/dashboard" replace />
    if (role === 'INVESTIGATOR') return <Navigate to="/investigator/dashboard" replace />
    return <Navigate to="/login" replace />
  }

  return children
}