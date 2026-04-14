import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import ProtectedRoute from './components/layout/ProtectedRoute'
import AuthPage from './pages/auth/AuthPage'
import ClientDashboard from './pages/client/Dashboard'
import NewClaim from './pages/client/NewClaim'
import ClaimDetail from './pages/client/ClaimDetail'
import ClaimsPage from './pages/client/ClaimsPage'
import ProfilePage from './pages/client/ProfilePage'
import InvestigatorDashboard from './pages/investigator/Dashboard'
import ClaimReview from './pages/investigator/ClaimReview'
import InvestigatorHistory from './pages/investigator/InvestigatorHistory'
import InvestigatorStats from './pages/investigator/InvestigatorStats'
import InvestigatorProfile from './pages/investigator/InvestigatorProfile'
import ClientStats from './pages/client/ClientStats'
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<AuthPage />} />
        <Route path="/client/dashboard"  element={<ProtectedRoute allowedRole="CLIENT"><ClientDashboard /></ProtectedRoute>} />
        <Route path="/client/new-claim"  element={<ProtectedRoute allowedRole="CLIENT"><NewClaim /></ProtectedRoute>} />
        <Route path="/client/claims/:id" element={<ProtectedRoute allowedRole="CLIENT"><ClaimDetail /></ProtectedRoute>} />
        <Route path="/client/claims"     element={<ProtectedRoute allowedRole="CLIENT"><ClaimsPage /></ProtectedRoute>} />
        <Route path="/client/profile"    element={<ProtectedRoute allowedRole="CLIENT"><ProfilePage /></ProtectedRoute>} />
         <Route path="/client/stats" element={<ProtectedRoute allowedRole="CLIENT"><ClientStats /></ProtectedRoute>} />
         <Route path="/investigator/dashboard" element={<ProtectedRoute allowedRole="INVESTIGATOR"><InvestigatorDashboard /></ProtectedRoute>} />
        <Route path="/investigator/flagged"   element={<ProtectedRoute allowedRole="INVESTIGATOR"><InvestigatorDashboard /></ProtectedRoute>} />
        <Route path="/investigator/review/:id" element={<ProtectedRoute allowedRole="INVESTIGATOR"><ClaimReview /></ProtectedRoute>} />
        <Route path="/investigator/history"   element={<ProtectedRoute allowedRole="INVESTIGATOR"><InvestigatorHistory /></ProtectedRoute>} />
        <Route path="/investigator/stats"     element={<ProtectedRoute allowedRole="INVESTIGATOR"><InvestigatorStats /></ProtectedRoute>} />
        <Route path="/investigator/profile"   element={<ProtectedRoute allowedRole="INVESTIGATOR"><InvestigatorProfile /></ProtectedRoute>} />
        <Route path="/" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App