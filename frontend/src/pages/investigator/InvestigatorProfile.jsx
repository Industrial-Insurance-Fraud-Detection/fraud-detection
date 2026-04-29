import { useState } from 'react'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import { InvestigatorSidebar } from '../../components/layout/InvestigatorLayout'
import NotificationBell from '../../components/ui/NotificationBell'

/**
 * InvestigatorProfile
 * Feature 8:  GET /users/me       — view my profile
 * Feature 9:  PATCH /users/me     — update my profile
 * Feature 7:  POST /auth/change-password — change password
 * Feature 2:  Token refresh is handled by axios interceptor
 * Feature 3:  POST /auth/logout
 * Feature 4:  POST /auth/logout-all
 */
export default function InvestigatorProfile() {
  const { user, setAuth, accessToken, refreshToken, logout } = useAuthStore()
  const [dark, toggleDark] = useDarkMode()
  const [activeTab, setActiveTab] = useState('info')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState('')
  const [error, setError] = useState('')

  const [profileForm, setProfileForm] = useState({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    company: user?.company || '',
    phone: user?.phone || '',
  })
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  })

  const pageBg = dark ? '#0D1626' : '#F7F8FC'
  const cardBg = dark ? '#111C30' : 'white'
  const cardBorder = dark ? '#1E2D45' : '#EEF0F6'
  const textMain = dark ? 'white' : '#0F2347'
  const textSub = dark ? '#5A7A9A' : '#9CA3AF'
  const inputBg = dark ? '#0D1626' : '#F9FAFB'
  const inputBorder = dark ? '#1E2D45' : '#E5E7EB'

  const showMsg = (type, msg) => {
    if (type === 'success') { setSuccess(msg); setError('') }
    else { setError(msg); setSuccess('') }
    setTimeout(() => { setSuccess(''); setError('') }, 4000)
  }

  // Feature 9 — update profile
  const handleUpdateProfile = async (e) => {
    e.preventDefault()
    if (!profileForm.firstName.trim() || !profileForm.lastName.trim()) {
      showMsg('error', 'Prenom et nom sont obligatoires'); return
    }
    setLoading(true)
    try {
      const res = await api.patch('/users/me', profileForm)
      const updatedUser = res.data?.data ?? res.data
      setAuth(updatedUser, accessToken, refreshToken)
      showMsg('success', 'Profil mis a jour avec succes !')
    } catch (err) {
      const msg = err.response?.data?.message
      showMsg('error', Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la mise a jour')
    } finally { setLoading(false) }
  }

  // Feature 7 — change password
  const handleUpdatePassword = async (e) => {
    e.preventDefault()
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      showMsg('error', 'Les mots de passe ne correspondent pas'); return
    }
    if (passwordForm.newPassword.length < 8) {
      showMsg('error', 'Le mot de passe doit faire au moins 8 caracteres'); return
    }
    setLoading(true)
    try {
      await api.post('/auth/change-password', {
        currentPassword: passwordForm.currentPassword,
        newPassword: passwordForm.newPassword,
      })
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' })
      showMsg('success', 'Mot de passe modifie ! Reconnectez-vous.')
    } catch (err) {
      const msg = err.response?.data?.message
      showMsg('error', Array.isArray(msg) ? msg.join(', ') : msg || 'Mot de passe actuel incorrect')
    } finally { setLoading(false) }
  }

  // Feature 4 — logout all devices
  const handleLogoutAll = async () => {
    try {
      await api.post('/auth/logout-all')
    } catch { /* ignore */ }
    logout()
    window.location.href = '/login'
  }

  const inputStyle = {
    width: '100%', padding: '0.75rem 1rem',
    border: `1.5px solid ${inputBorder}`, borderRadius: 8,
    fontSize: '0.9rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
    outline: 'none', backgroundColor: inputBg, color: textMain,
    boxSizing: 'border-box',
  }
  const labelStyle = {
    display: 'block', fontSize: '0.74rem', fontWeight: 600,
    textTransform: 'uppercase', letterSpacing: '0.06em',
    color: textSub, marginBottom: '0.4rem',
    fontFamily: 'Helvetica Neue, Arial, sans-serif',
  }

  const fullName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Investigateur'
  const initial = (user?.firstName?.[0] ?? 'I').toUpperCase()

  const tabs = [
    { key: 'info', label: 'Informations', icon: '👤' },
    { key: 'password', label: 'Mot de passe', icon: '🔒' },
    { key: 'security', label: 'Securite', icon: '🛡' },
  ]

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif', transition: 'background 0.3s' }}>
      <InvestigatorSidebar dark={dark} />
      <div style={{ marginLeft: 240, flex: 1, padding: '2rem' }}>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <p style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.14em', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Compte</p>
            <h1 style={{ fontSize: '1.9rem', color: textMain, fontWeight: 400 }}>Mon <strong>profil</strong></h1>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <NotificationBell dark={dark} />
            <button onClick={toggleDark} style={{ padding: '0.55rem 1rem', border: `1.5px solid ${cardBorder}`, borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: cardBg, color: textSub }}>
              {dark ? '☀ Mode clair' : '🌙 Mode sombre'}
            </button>
          </div>
        </div>

        {success && <div style={{ backgroundColor: '#F0FAF4', border: '1px solid #B8E4CA', borderRadius: 8, padding: '0.75rem 1rem', color: '#1A7A4A', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>✓ {success}</div>}
        {error && <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 8, padding: '0.75rem 1rem', color: '#C0392B', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>⚠ {error}</div>}

        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '1.5rem' }}>
          {/* Left */}
          <div>
            <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '2rem', textAlign: 'center', marginBottom: '1rem' }}>
              <div style={{ width: 96, height: 96, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, fontSize: '2.2rem', margin: '0 auto 1rem', border: '3px solid #C9A84C' }}>
                {initial}
              </div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.25rem' }}>{fullName}</div>
              <div style={{ fontSize: '0.78rem', color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.25rem' }}>{user?.email}</div>
              <div style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>{user?.company || 'Sans organisation'}</div>
              <span style={{ padding: '0.3rem 0.8rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, backgroundColor: '#EBF5FB', color: '#1A5276', border: '1px solid #AED6F1', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                INVESTIGATEUR
              </span>
            </div>

            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, overflow: 'hidden' }}>
              {tabs.map((tab, i) => (
                <div key={tab.key} onClick={() => setActiveTab(tab.key)}
                  style={{ padding: '0.85rem 1.25rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.75rem', backgroundColor: activeTab === tab.key ? 'rgba(201,168,76,0.1)' : 'transparent', borderLeft: activeTab === tab.key ? '3px solid #C9A84C' : '3px solid transparent', borderBottom: i < tabs.length - 1 ? `1px solid ${cardBorder}` : 'none' }}>
                  <span>{tab.icon}</span>
                  <span style={{ fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', color: activeTab === tab.key ? textMain : textSub, fontWeight: activeTab === tab.key ? 600 : 400 }}>{tab.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Right */}
          <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '2rem' }}>

            {/* Feature 9 — update profile */}
            {activeTab === 'info' && (
              <form onSubmit={handleUpdateProfile}>
                <h2 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Informations personnelles</h2>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                  <div>
                    <label style={labelStyle}>Prenom *</label>
                    <input value={profileForm.firstName} onChange={e => setProfileForm({ ...profileForm, firstName: e.target.value })}
                      style={inputStyle} placeholder="Votre prenom"
                      onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                  </div>
                  <div>
                    <label style={labelStyle}>Nom *</label>
                    <input value={profileForm.lastName} onChange={e => setProfileForm({ ...profileForm, lastName: e.target.value })}
                      style={inputStyle} placeholder="Votre nom"
                      onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                  <div>
                    <label style={labelStyle}>Organisation</label>
                    <input value={profileForm.company} onChange={e => setProfileForm({ ...profileForm, company: e.target.value })}
                      style={inputStyle} placeholder="Compagnie d'assurance"
                      onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                  </div>
                  <div>
                    <label style={labelStyle}>Telephone</label>
                    <input value={profileForm.phone} onChange={e => setProfileForm({ ...profileForm, phone: e.target.value })}
                      style={inputStyle} placeholder="+213550000001"
                      onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                  </div>
                </div>
                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={labelStyle}>Adresse email</label>
                  <input value={user?.email || ''} disabled style={{ ...inputStyle, opacity: 0.6, cursor: 'not-allowed' }} />
                  <p style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.3rem' }}>L'email ne peut pas etre modifie</p>
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button type="submit" disabled={loading}
                    style={{ padding: '0.75rem 1.75rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1 }}>
                    {loading ? 'Enregistrement...' : 'Enregistrer les modifications'}
                  </button>
                </div>
              </form>
            )}

            {/* Feature 7 — change password */}
            {activeTab === 'password' && (
              <form onSubmit={handleUpdatePassword}>
                <h2 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Modifier le mot de passe</h2>
                <p style={{ color: textSub, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '2rem' }}>Minimum 8 caracteres avec majuscule, minuscule et chiffre</p>
                {[
                  { label: 'Mot de passe actuel *', key: 'currentPassword', placeholder: 'Votre mot de passe actuel' },
                  { label: 'Nouveau mot de passe *', key: 'newPassword', placeholder: 'Ex: MonMotDePasse1' },
                  { label: 'Confirmer *', key: 'confirmPassword', placeholder: 'Repetez le nouveau mot de passe' },
                ].map(f => (
                  <div key={f.key} style={{ marginBottom: '1rem' }}>
                    <label style={labelStyle}>{f.label}</label>
                    <input type="password" placeholder={f.placeholder} value={passwordForm[f.key]}
                      onChange={e => setPasswordForm({ ...passwordForm, [f.key]: e.target.value })}
                      style={inputStyle}
                      onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                  </div>
                ))}
                {passwordForm.newPassword && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div style={{ height: 6, backgroundColor: dark ? '#1E2D45' : '#F3F4F6', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', borderRadius: 3, transition: 'width 0.3s, background 0.3s',
                        width: passwordForm.newPassword.length < 6 ? '25%' : passwordForm.newPassword.length < 10 ? '60%' : '100%',
                        backgroundColor: passwordForm.newPassword.length < 6 ? '#C0392B' : passwordForm.newPassword.length < 10 ? '#F39C12' : '#1A7A4A'
                      }} />
                    </div>
                    <div style={{
                      fontSize: '0.68rem', marginTop: '0.25rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
                      color: passwordForm.newPassword.length < 6 ? '#C0392B' : passwordForm.newPassword.length < 10 ? '#F39C12' : '#1A7A4A'
                    }}>
                      {passwordForm.newPassword.length < 6 ? 'Faible' : passwordForm.newPassword.length < 10 ? 'Moyen' : 'Fort'}
                    </div>
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button type="submit" disabled={loading || !passwordForm.currentPassword || !passwordForm.newPassword}
                    style={{ padding: '0.75rem 1.75rem', background: !passwordForm.currentPassword || !passwordForm.newPassword ? '#E5E7EB' : 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: !passwordForm.currentPassword || !passwordForm.newPassword ? '#9CA3AF' : 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                    {loading ? 'Modification...' : 'Modifier le mot de passe'}
                  </button>
                </div>
              </form>
            )}

            {/* Feature 3, 4 + security info */}
            {activeTab === 'security' && (
              <div>
                <h2 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>Securite du compte</h2>
                <p style={{ color: textSub, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '2rem' }}>Informations et actions de securite</p>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '2rem' }}>
                  {[
                    { icon: '🔒', label: 'JWT Securise', desc: 'Token expire dans 15 minutes' },
                    { icon: '🛡', label: 'bcrypt', desc: 'Mot de passe chiffre' },
                    { icon: '🔐', label: 'CORS protege', desc: 'Requetes cross-origin filtrees' },
                    { icon: '📋', label: 'Validation donnees', desc: 'Toutes les entrees sont validees' },
                  ].map(item => (
                    <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '1rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 10, border: `1px solid ${cardBorder}` }}>
                      <span style={{ fontSize: '1.4rem' }}>{item.icon}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: '0.88rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{item.label}</div>
                        <div style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: 2 }}>{item.desc}</div>
                      </div>
                      <span style={{ padding: '0.25rem 0.75rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, backgroundColor: '#F0FAF4', color: '#1A7A4A', border: '1px solid #B8E4CA' }}>Actif</span>
                    </div>
                  ))}
                </div>

                {/* Feature 4 — logout all */}
                <div style={{ padding: '1.25rem', backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 10, marginBottom: '1rem' }}>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.3rem' }}>
                    ⚠ Deconnexion de tous les appareils
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#C0392B', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem', lineHeight: 1.5 }}>
                    Cette action invalidera tous vos tokens actifs et vous deconnectera de chaque appareil.
                  </div>
                  <button onClick={handleLogoutAll}
                    style={{ padding: '0.6rem 1.2rem', background: 'linear-gradient(135deg, #C0392B, #E74C3C)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                    Deconnecter tous les appareils
                  </button>
                </div>

                <div style={{ padding: '1rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 10, border: `1px solid ${cardBorder}` }}>
                  <div style={{ fontSize: '0.82rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.75rem' }}>Informations du compte</div>
                  {[
                    ['ID', user?.id?.substring(0, 8) + '...'],
                    ['Role', user?.role],
                    ['Membre depuis', user?.createdAt ? new Date(user.createdAt).toLocaleDateString('fr-FR') : '-'],
                  ].map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: `1px solid ${cardBorder}` }}>
                      <span style={{ fontSize: '0.78rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{k}</span>
                      <span style={{ fontSize: '0.78rem', color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 500 }}>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}