import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import useAuthStore from '../../store/auth.store'
import AuthLeftPanel from './AuthLeftPanel'

// ── Password strength checker ─────────────────────────────────────────────────
function PasswordStrength({ password }) {
  if (!password) return null
  const checks = {
    len: password.length >= 8,
    upper: /[A-Z]/.test(password),
    lower: /[a-z]/.test(password),
    digit: /[0-9]/.test(password),
  }
  const passed = Object.values(checks).filter(Boolean).length
  const bar = ['#E5E7EB', '#C0392B', '#E67E22', '#F39C12', '#1A7A4A']
  const lbl = ['', 'Très faible', 'Faible', 'Moyen', 'Fort']
  return (
    <div style={{ marginTop: '0.55rem' }}>
      {/* Segmented bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: '0.45rem' }}>
        {[1, 2, 3, 4].map(i => (
          <div key={i} style={{
            flex: 1, height: 4, borderRadius: 3,
            backgroundColor: i <= passed ? bar[passed] : '#F3F4F6',
            transition: 'background-color 0.35s',
          }} />
        ))}
      </div>
      {/* Strength label */}
      <div style={{ fontSize: '0.7rem', color: bar[passed], fontFamily: 'Helvetica Neue,Arial,sans-serif', fontWeight: 700, marginBottom: '0.45rem' }}>
        {lbl[passed]}
      </div>
      {/* Checklist */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem 1rem' }}>
        {[
          [checks.len, '8 caractères min.'],
          [checks.upper, 'Majuscule (A–Z)'],
          [checks.lower, 'Minuscule (a–z)'],
          [checks.digit, 'Chiffre (0–9)'],
        ].map(([ok, txt]) => (
          <div key={txt} style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.68rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', color: ok ? '#1A7A4A' : '#9CA3AF', transition: 'color 0.2s' }}>
            <span style={{ fontWeight: 700, fontSize: '0.7rem' }}>{ok ? '✓' : '○'}</span>
            {txt}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function AuthPage() {
  const navigate = useNavigate()
  const { setAuth } = useAuthStore()
  const [tab, setTab] = useState('login')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [showPass, setShowPass] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)

  const [loginForm, setLoginForm] = useState({ email: '', password: '' })
  const [registerForm, setRegisterForm] = useState({
    firstName: '', lastName: '', company: '', email: '',
    password: '', confirmPassword: '', phone: '', wilaya: '',
  })

  // ── Login ─────────────────────────────────────────────────────────────────
  const handleLogin = async (e) => {
    e.preventDefault()
    setError(''); setSuccess('')
    if (!loginForm.email || !loginForm.password) { setError('Veuillez remplir tous les champs.'); return }
    setLoading(true)
    try {
      const res = await api.post('/auth/login', { email: loginForm.email, password: loginForm.password })
      const { accessToken, refreshToken, user } = res.data?.data ?? res.data
      setAuth(user, accessToken, refreshToken)
      navigate(user.role === 'CLIENT' ? '/client/dashboard' : '/investigator/dashboard')
    } catch (err) {
      const msg = err.response?.data?.message
      setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Email ou mot de passe incorrect.')
    } finally { setLoading(false) }
  }

  // ── Register ──────────────────────────────────────────────────────────────
  const handleRegister = async (e) => {
    e.preventDefault()
    setError(''); setSuccess('')
    if (!registerForm.firstName || !registerForm.lastName || !registerForm.email || !registerForm.password) {
      setError('Veuillez remplir tous les champs obligatoires.'); return
    }
    if (registerForm.password !== registerForm.confirmPassword) { setError('Les mots de passe ne correspondent pas.'); return }
    if (registerForm.password.length < 8) { setError('Le mot de passe doit contenir au moins 8 caractères.'); return }
    setLoading(true)
    try {
      await api.post('/auth/register', {
        firstName: registerForm.firstName, lastName: registerForm.lastName,
        company: registerForm.company || undefined, email: registerForm.email,
        password: registerForm.password,
        phone: registerForm.phone || undefined, wilaya: registerForm.wilaya || undefined,
      })
      setSuccess('Compte créé avec succès ! Connectez-vous.')
      setTab('login')
      setRegisterForm({ firstName: '', lastName: '', company: '', email: '', password: '', confirmPassword: '', phone: '', wilaya: '' })
    } catch (err) {
      const msg = err.response?.data?.message
      setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la création du compte.')
    } finally { setLoading(false) }
  }

  const switchTab = (t) => { setTab(t); setError(''); setSuccess('') }

  // ── Shared styles ─────────────────────────────────────────────────────────
  const inp = {
    width: '100%', padding: '0.75rem 1rem',
    border: '1.5px solid #E5E7EB', borderRadius: 8,
    fontSize: '0.88rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
    outline: 'none', backgroundColor: '#F9FAFB', boxSizing: 'border-box',
    transition: 'border-color 0.2s, box-shadow 0.2s', color: '#0F2347',
  }
  const lbl = {
    display: 'block', fontSize: '0.7rem', fontWeight: 600,
    textTransform: 'uppercase', letterSpacing: '0.08em',
    color: '#6B7280', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
  }
  const onFocus = e => { e.target.style.borderColor = '#C9A84C'; e.target.style.boxShadow = '0 0 0 3px rgba(201,168,76,0.12)' }
  const onBlur = e => { e.target.style.borderColor = '#E5E7EB'; e.target.style.boxShadow = 'none' }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', fontFamily: 'Georgia,serif' }}>

      <AuthLeftPanel />

      {/* ── RIGHT PANEL ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '2.5rem 4rem', backgroundColor: 'white', overflowY: 'auto' }}>
        <div style={{ maxWidth: 420, width: '100%', margin: '0 auto' }}>

          <p style={{ fontSize: '0.68rem', letterSpacing: '0.16em', textTransform: 'uppercase', color: '#9CA3AF', marginBottom: '0.35rem', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>
            {tab === 'login' ? 'Bienvenue' : 'Nouveau compte'}
          </p>
          <h2 style={{ color: '#0F2347', fontSize: '1.85rem', fontWeight: 400, marginBottom: '0.35rem', letterSpacing: '-0.02em' }}>
            {tab === 'login'
              ? <><strong>Connexion</strong> à votre espace</>
              : <><strong>Créer</strong> un compte</>}
          </h2>
          <p style={{ color: '#9CA3AF', fontSize: '0.83rem', marginBottom: '1.5rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', lineHeight: 1.55 }}>
            {tab === 'login'
              ? 'Accès sécurisé à votre tableau de bord sinistres.'
              : 'Créez votre compte client pour soumettre et suivre vos sinistres.'}
          </p>

          {/* Tabs */}
          <div style={{ display: 'flex', borderBottom: '2px solid #F3F4F6', marginBottom: '1.5rem' }}>
            {[['login', 'Connexion'], ['register', 'Inscription']].map(([t, l]) => (
              <button key={t} onClick={() => switchTab(t)} style={{
                padding: '0.55rem 1.4rem', fontSize: '0.85rem',
                fontFamily: 'Helvetica Neue,Arial,sans-serif',
                border: 'none', background: 'transparent', cursor: 'pointer',
                borderBottom: tab === t ? '2px solid #C9A84C' : '2px solid transparent',
                marginBottom: -2, color: tab === t ? '#0F2347' : '#9CA3AF',
                fontWeight: tab === t ? 600 : 400, transition: 'all 0.2s',
              }}>{l}</button>
            ))}
          </div>

          {/* Alerts */}
          {error && (
            <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 8, padding: '0.65rem 0.9rem', color: '#C0392B', fontSize: '0.81rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginBottom: '1rem', display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
              <span style={{ flexShrink: 0 }}>⚠</span> {error}
            </div>
          )}
          {success && (
            <div style={{ backgroundColor: '#F0FAF4', border: '1px solid #B8E4CA', borderRadius: 8, padding: '0.65rem 0.9rem', color: '#1A7A4A', fontSize: '0.81rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginBottom: '1rem', display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
              <span style={{ flexShrink: 0 }}>✓</span> {success}
            </div>
          )}

          {/* ── LOGIN ── */}
          {tab === 'login' && (
            <form onSubmit={handleLogin}>
              <div style={{ marginBottom: '1rem' }}>
                <label style={lbl}>Adresse email</label>
                <input type="email" placeholder="votre@email.com" value={loginForm.email}
                  onChange={e => setLoginForm({ ...loginForm, email: e.target.value })}
                  style={inp} onFocus={onFocus} onBlur={onBlur} />
              </div>

              <div style={{ marginBottom: '0.5rem' }}>
                <label style={lbl}>Mot de passe</label>
                <div style={{ position: 'relative' }}>
                  <input type={showPass ? 'text' : 'password'} placeholder="••••••••" value={loginForm.password}
                    onChange={e => setLoginForm({ ...loginForm, password: e.target.value })}
                    style={{ ...inp, paddingRight: '2.8rem' }} onFocus={onFocus} onBlur={onBlur} />
                  <button type="button" onClick={() => setShowPass(!showPass)} style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#9CA3AF', padding: 0, fontSize: '1rem' }}>
                    {showPass ? '🙈' : '👁'}
                  </button>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1.25rem' }}>
                <span onClick={() => navigate('/forgot-password')} style={{ fontSize: '0.76rem', color: '#C9A84C', cursor: 'pointer', fontFamily: 'Helvetica Neue,Arial,sans-serif', fontWeight: 600 }}>
                  Mot de passe oublié ?
                </span>
              </div>

              <button type="submit" disabled={loading} style={{
                width: '100%', padding: '0.82rem',
                background: loading ? '#9CA3AF' : 'linear-gradient(135deg, #0F2347 0%, #1A3A6B 100%)',
                color: 'white', border: 'none', borderRadius: 8,
                fontSize: '0.86rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
                fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase',
                cursor: loading ? 'not-allowed' : 'pointer',
                boxShadow: loading ? 'none' : '0 4px 18px rgba(15,35,71,0.28)',
                transition: 'all 0.2s',
              }}>
                {loading ? 'Connexion…' : 'Se connecter →'}
              </button>
            </form>
          )}

          {/* ── REGISTER ── */}
          {tab === 'register' && (
            <form onSubmit={handleRegister}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.7rem', marginBottom: '0.75rem' }}>
                {[['firstName', 'Prénom *', 'Votre prénom'], ['lastName', 'Nom *', 'Votre nom']].map(([k, l, p]) => (
                  <div key={k}>
                    <label style={lbl}>{l}</label>
                    <input type="text" placeholder={p} value={registerForm[k]}
                      onChange={e => setRegisterForm({ ...registerForm, [k]: e.target.value })}
                      style={inp} onFocus={onFocus} onBlur={onBlur} />
                  </div>
                ))}
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={lbl}>Entreprise</label>
                <input type="text" placeholder="Nom de votre entreprise (optionnel)" value={registerForm.company}
                  onChange={e => setRegisterForm({ ...registerForm, company: e.target.value })}
                  style={inp} onFocus={onFocus} onBlur={onBlur} />
              </div>

              <div style={{ marginBottom: '0.75rem' }}>
                <label style={lbl}>Adresse email *</label>
                <input type="email" placeholder="votre@email.com" value={registerForm.email}
                  onChange={e => setRegisterForm({ ...registerForm, email: e.target.value })}
                  style={inp} onFocus={onFocus} onBlur={onBlur} />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.7rem', marginBottom: '0.75rem' }}>
                <div>
                  <label style={lbl}>Téléphone</label>
                  <input type="tel" placeholder="0550000001" value={registerForm.phone}
                    onChange={e => setRegisterForm({ ...registerForm, phone: e.target.value })}
                    style={inp} onFocus={onFocus} onBlur={onBlur} />
                </div>
                <div>
                  <label style={lbl}>Wilaya</label>
                  <input type="text" placeholder="Ex: Boumerdès" value={registerForm.wilaya}
                    onChange={e => setRegisterForm({ ...registerForm, wilaya: e.target.value })}
                    style={inp} onFocus={onFocus} onBlur={onBlur} />
                </div>
              </div>

              {/* Password with strength */}
              <div style={{ marginBottom: '0.75rem' }}>
                <label style={lbl}>Mot de passe *</label>
                <div style={{ position: 'relative' }}>
                  <input type={showPass ? 'text' : 'password'} placeholder="Créez un mot de passe sécurisé" value={registerForm.password}
                    onChange={e => setRegisterForm({ ...registerForm, password: e.target.value })}
                    style={{ ...inp, paddingRight: '2.8rem' }} onFocus={onFocus} onBlur={onBlur} />
                  <button type="button" onClick={() => setShowPass(!showPass)} style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#9CA3AF', padding: 0, fontSize: '1rem' }}>
                    {showPass ? '🙈' : '👁'}
                  </button>
                </div>
                <PasswordStrength password={registerForm.password} />
              </div>

              {/* Confirm password */}
              <div style={{ marginBottom: '1rem' }}>
                <label style={lbl}>Confirmer le mot de passe *</label>
                <div style={{ position: 'relative' }}>
                  <input type={showConfirm ? 'text' : 'password'} placeholder="Répétez le mot de passe"
                    value={registerForm.confirmPassword}
                    onChange={e => setRegisterForm({ ...registerForm, confirmPassword: e.target.value })}
                    style={{
                      ...inp, paddingRight: '2.8rem',
                      borderColor: registerForm.confirmPassword && registerForm.confirmPassword !== registerForm.password ? '#C0392B' : '#E5E7EB',
                    }}
                    onFocus={onFocus} onBlur={onBlur} />
                  <button type="button" onClick={() => setShowConfirm(!showConfirm)} style={{ position: 'absolute', right: '0.75rem', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#9CA3AF', padding: 0, fontSize: '1rem' }}>
                    {showConfirm ? '🙈' : '👁'}
                  </button>
                </div>
                {registerForm.confirmPassword && registerForm.confirmPassword !== registerForm.password && (
                  <div style={{ fontSize: '0.68rem', color: '#C0392B', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginTop: '0.25rem' }}>✗ Les mots de passe ne correspondent pas</div>
                )}
                {registerForm.confirmPassword && registerForm.confirmPassword === registerForm.password && (
                  <div style={{ fontSize: '0.68rem', color: '#1A7A4A', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginTop: '0.25rem' }}>✓ Les mots de passe correspondent</div>
                )}
              </div>

              <button type="submit" disabled={loading} style={{
                width: '100%', padding: '0.82rem',
                background: loading ? '#9CA3AF' : 'linear-gradient(135deg, #0F2347 0%, #1A3A6B 100%)',
                color: 'white', border: 'none', borderRadius: 8,
                fontSize: '0.86rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
                fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase',
                cursor: loading ? 'not-allowed' : 'pointer',
                boxShadow: loading ? 'none' : '0 4px 18px rgba(15,35,71,0.28)',
              }}>
                {loading ? 'Création du compte…' : 'Créer mon compte →'}
              </button>
            </form>
          )}

          <p style={{ textAlign: 'center', marginTop: '1.1rem', fontSize: '0.79rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>
            {tab === 'login' ? 'Pas encore de compte ? ' : 'Déjà un compte ? '}
            <span onClick={() => switchTab(tab === 'login' ? 'register' : 'login')} style={{ color: '#C9A84C', fontWeight: 600, cursor: 'pointer' }}>
              {tab === 'login' ? "S'inscrire" : 'Se connecter'}
            </span>
          </p>
        </div>
      </div>
    </div>
  )
}