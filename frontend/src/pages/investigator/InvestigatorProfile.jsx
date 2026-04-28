import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import useAuthStore from '../../store/auth.store'
import api from '../../api/axios'
import { useDarkMode } from '../../components/layout/Sidebar'
import NotificationBell from '../../components/ui/NotificationBell'

/**
 * InvestigatorProfile
 *
 * FIXED:
 *   1. setAuth(user, accessToken, refreshToken) — 3 args, not (data, token, role)
 *   2. user.firstName / user.lastName — not user.fullName
 *   3. Password change uses POST /auth/change-password (same as client),
 *      not PATCH /users/me/password (that endpoint doesn't exist)
 */

function InvestigatorSidebar({ dark }) {
  const navigate = useNavigate()
  const { logout, user } = useAuthStore()
  const [collapsed, setCollapsed] = useState(false)
  const items = [
    { key: '/investigator/dashboard', label: 'Tableau de bord', icon: '▦' },
    { key: '/investigator/flagged', label: 'Dossiers a traiter', icon: '⚑' },
    { key: '/investigator/history', label: 'Historique', icon: '≡' },
    { key: '/investigator/stats', label: 'Statistiques', icon: '◑' },
    { key: '/investigator/profile', label: 'Mon profil', icon: '👤' },
  ]
  const bg = dark ? '#0A1628' : '#0F2347'
  const border = dark ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.08)'
  const width = collapsed ? 64 : 240
  const active = window.location.pathname
  // FIXED: derive from firstName/lastName
  const invName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Investigateur'
  const initial = user?.firstName?.[0]?.toUpperCase() || 'I'

  return (
    <div style={{ width, minHeight: '100vh', backgroundColor: bg, display: 'flex', flexDirection: 'column', position: 'fixed', left: 0, top: 0, zIndex: 100, transition: 'width 0.25s', overflow: 'hidden', boxShadow: '4px 0 24px rgba(0,0,0,0.15)' }}>
      <div style={{ padding: collapsed ? '1.5rem 0.75rem' : '1.5rem', borderBottom: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {!collapsed ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347' }}>F</div>
              <div>
                <div style={{ color: 'white', fontWeight: 700, fontSize: '0.92rem' }}>FraudGuard AI</div>
                <div style={{ color: '#C9A84C', fontSize: '0.58rem', letterSpacing: '0.12em', textTransform: 'uppercase' }}>Espace Investigateur</div>
              </div>
            </div>
            <button onClick={() => setCollapsed(true)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.3)', fontSize: '1rem' }}>←</button>
          </>
        ) : (
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold', color: '#0F2347', margin: '0 auto', cursor: 'pointer' }} onClick={() => setCollapsed(false)}>F</div>
        )}
      </div>
      {!collapsed && (
        <div style={{ padding: '1rem 1.5rem', borderBottom: `1px solid ${border}` }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 38, height: 38, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700 }}>
              {initial}
            </div>
            <div>
              <div style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>{invName}</div>
              <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.68rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Investigateur senior</div>
            </div>
          </div>
        </div>
      )}
      <nav style={{ flex: 1, padding: '0.75rem 0' }}>
        {items.map(item => {
          const isActive = active === item.key
          return (
            <div key={item.key} onClick={() => navigate(item.key)}
              style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', padding: collapsed ? '0.75rem' : '0.75rem 1.5rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', backgroundColor: isActive ? 'rgba(201,168,76,0.12)' : 'transparent', borderLeft: isActive ? '3px solid #C9A84C' : '3px solid transparent', transition: 'all 0.18s' }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)' }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.backgroundColor = 'transparent' }}>
              <span style={{ color: isActive ? '#C9A84C' : 'rgba(255,255,255,0.45)', width: 20, textAlign: 'center' }}>{item.icon}</span>
              {!collapsed && <span style={{ color: isActive ? 'white' : 'rgba(255,255,255,0.55)', fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: isActive ? 600 : 400 }}>{item.label}</span>}
            </div>
          )
        })}
      </nav>
      <div style={{ padding: collapsed ? '0.75rem' : '1rem 1.5rem', borderTop: `1px solid ${border}` }}>
        <div onClick={() => { logout(); window.location.href = '/login' }} style={{ display: 'flex', alignItems: 'center', gap: collapsed ? 0 : '0.75rem', justifyContent: collapsed ? 'center' : 'flex-start', cursor: 'pointer', padding: '0.4rem', borderRadius: 6 }}>
          <span style={{ color: 'rgba(255,255,255,0.35)' }}>↩</span>
          {!collapsed && <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>Deconnexion</span>}
        </div>
      </div>
    </div>
  )
}

export default function InvestigatorProfile() {
  // FIXED: destructure correctly — no `token` or `role` in store API
  const { user, setAuth, accessToken, refreshToken } = useAuthStore()
  const [dark, toggleDark] = useDarkMode()
  const [activeTab, setActiveTab] = useState('info')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState('')
  const [error, setError] = useState('')
  const fileRef = useRef()

  // FIXED: use firstName/lastName, not fullName
  const [profileForm, setProfileForm] = useState({
    firstName: user?.firstName || '',
    lastName: user?.lastName || '',
    company: user?.company || '',
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

  const handleUpdateProfile = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await api.patch('/users/me', {
        firstName: profileForm.firstName,
        lastName: profileForm.lastName,
        company: profileForm.company,
      })
      // TransformInterceptor wraps: { success, data: updatedUser }
      const updatedUser = res.data?.data ?? res.data
      // FIXED: pass all 3 args — keeps refreshToken alive
      setAuth(updatedUser, accessToken, refreshToken)
      showMsg('success', 'Profil mis a jour !')
    } catch (err) {
      showMsg('error', err.response?.data?.message || 'Erreur')
    } finally { setLoading(false) }
  }

  const handleUpdatePassword = async (e) => {
    e.preventDefault()
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      showMsg('error', 'Mots de passe differents')
      return
    }
    if (passwordForm.newPassword.length < 8) {
      showMsg('error', 'Le mot de passe doit faire au moins 8 caracteres')
      return
    }
    setLoading(true)
    try {
      // FIXED: correct endpoint — same as client (PATCH /users/me/password doesn't exist)
      await api.post('/auth/change-password', {
        currentPassword: passwordForm.currentPassword,
        newPassword: passwordForm.newPassword,
      })
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' })
      showMsg('success', 'Mot de passe modifie !')
    } catch (err) {
      showMsg('error', err.response?.data?.message || 'Mot de passe actuel incorrect')
    } finally { setLoading(false) }
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

  // FIXED: derive display name from firstName + lastName
  const fullName = `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || 'Investigateur'
  const initial = user?.firstName?.[0]?.toUpperCase() || 'I'

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: pageBg, fontFamily: 'Georgia, serif' }}>
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

          {/* Left card */}
          <div>
            <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '2rem', textAlign: 'center', marginBottom: '1rem' }}>
              <div style={{ position: 'relative', display: 'inline-block', marginBottom: '1rem' }}>
                <div style={{ width: 96, height: 96, borderRadius: '50%', background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#0F2347', fontWeight: 700, fontSize: '2.2rem', border: '3px solid #C9A84C' }}>
                  {initial}
                </div>
                <button onClick={() => fileRef.current.click()} style={{ position: 'absolute', bottom: 0, right: 0, width: 28, height: 28, borderRadius: '50%', backgroundColor: '#0F2347', border: `2px solid ${cardBg}`, cursor: 'pointer', fontSize: '0.8rem' }}>📷</button>
                <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} />
              </div>
              {/* FIXED: display firstName + lastName */}
              <div style={{ fontSize: '1.1rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '0.25rem' }}>{fullName}</div>
              <div style={{ fontSize: '0.78rem', color: '#C9A84C', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>{user?.email}</div>
              <span style={{ padding: '0.3rem 0.8rem', borderRadius: 20, fontSize: '0.72rem', fontWeight: 600, backgroundColor: '#EBF5FB', color: '#1A5276', border: '1px solid #AED6F1', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>INVESTIGATOR</span>
            </div>

            <div style={{ backgroundColor: cardBg, borderRadius: 12, border: `1px solid ${cardBorder}`, overflow: 'hidden' }}>
              {[
                { key: 'info', label: 'Informations', icon: '👤' },
                { key: 'password', label: 'Mot de passe', icon: '🔒' },
                { key: 'security', label: 'Securite', icon: '🛡' },
              ].map((tab, i, arr) => (
                <div key={tab.key} onClick={() => setActiveTab(tab.key)}
                  style={{ padding: '0.85rem 1.25rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.75rem', backgroundColor: activeTab === tab.key ? 'rgba(201,168,76,0.1)' : 'transparent', borderLeft: activeTab === tab.key ? '3px solid #C9A84C' : '3px solid transparent', borderBottom: i < arr.length - 1 ? `1px solid ${cardBorder}` : 'none' }}>
                  <span>{tab.icon}</span>
                  <span style={{ fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', color: activeTab === tab.key ? textMain : textSub, fontWeight: activeTab === tab.key ? 600 : 400 }}>{tab.label}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Right content */}
          <div style={{ backgroundColor: cardBg, borderRadius: 14, border: `1px solid ${cardBorder}`, padding: '2rem' }}>

            {activeTab === 'info' && (
              <form onSubmit={handleUpdateProfile}>
                <h2 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Informations personnelles</h2>

                {/* FIXED: firstName + lastName separately */}
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

                <div style={{ marginBottom: '1rem' }}>
                  <label style={labelStyle}>Organisation</label>
                  <input value={profileForm.company} onChange={e => setProfileForm({ ...profileForm, company: e.target.value })}
                    style={inputStyle} placeholder="Compagnie d'assurance"
                    onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                </div>

                <div style={{ marginBottom: '1.5rem' }}>
                  <label style={labelStyle}>Email</label>
                  <input value={user?.email || ''} disabled style={{ ...inputStyle, opacity: 0.6, cursor: 'not-allowed' }} />
                  <p style={{ fontSize: '0.72rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginTop: '0.3rem' }}>L'email ne peut pas etre modifie</p>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button type="submit" disabled={loading}
                    style={{ padding: '0.75rem 1.75rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', opacity: loading ? 0.7 : 1 }}>
                    {loading ? 'Enregistrement...' : 'Enregistrer'}
                  </button>
                </div>
              </form>
            )}

            {activeTab === 'password' && (
              <form onSubmit={handleUpdatePassword}>
                <h2 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Modifier le mot de passe</h2>
                {[
                  { label: 'Mot de passe actuel *', key: 'currentPassword' },
                  { label: 'Nouveau mot de passe *', key: 'newPassword' },
                  { label: 'Confirmer le nouveau mot de passe', key: 'confirmPassword' },
                ].map(f => (
                  <div key={f.key} style={{ marginBottom: '1rem' }}>
                    <label style={labelStyle}>{f.label}</label>
                    <input type="password" value={passwordForm[f.key]}
                      onChange={e => setPasswordForm({ ...passwordForm, [f.key]: e.target.value })}
                      style={inputStyle}
                      onFocus={e => e.target.style.borderColor = '#C9A84C'} onBlur={e => e.target.style.borderColor = inputBorder} />
                  </div>
                ))}
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.5rem' }}>
                  <button type="submit" disabled={loading}
                    style={{ padding: '0.75rem 1.75rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.85rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer' }}>
                    {loading ? 'Modification...' : 'Modifier'}
                  </button>
                </div>
              </form>
            )}

            {activeTab === 'security' && (
              <div>
                <h2 style={{ color: textMain, fontSize: '1.1rem', fontWeight: 600, fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.5rem' }}>Securite du compte</h2>
                {[
                  { icon: '🔒', label: 'JWT Securise', desc: 'Token expire dans 15 minutes' },
                  { icon: '🛡', label: 'bcrypt', desc: 'Chiffrement 10 rounds' },
                  { icon: '🔐', label: 'CORS protege', desc: 'Requetes filtrees' },
                  { icon: '📋', label: 'Validation donnees', desc: 'Entrees validees' },
                ].map(item => (
                  <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.75rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 8, border: `1px solid ${cardBorder}`, marginBottom: '0.75rem' }}>
                    <span style={{ fontSize: '1.2rem' }}>{item.icon}</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: textMain, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{item.label}</div>
                      <div style={{ fontSize: '0.75rem', color: textSub, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{item.desc}</div>
                    </div>
                    <span style={{ padding: '0.2rem 0.6rem', borderRadius: 20, fontSize: '0.68rem', fontWeight: 600, backgroundColor: '#F0FAF4', color: '#1A7A4A', border: '1px solid #B8E4CA' }}>Actif</span>
                  </div>
                ))}

                <div style={{ padding: '1rem', backgroundColor: dark ? '#0D1626' : '#F7F8FC', borderRadius: 10, border: `1px solid ${cardBorder}`, marginTop: '1rem' }}>
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