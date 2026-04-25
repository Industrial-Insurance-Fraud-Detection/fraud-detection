import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import useAuthStore from '../../store/auth.store'

export default function AuthPage() {
  const navigate = useNavigate()
  const { setAuth } = useAuthStore()
  const [tab, setTab] = useState('login')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const [loginForm, setLoginForm] = useState({ email: '', password: '' })
  const [registerForm, setRegisterForm] = useState({
    firstName: '', lastName: '', company: '', email: '', password: '', confirmPassword: '',
  })

  // ── Login ──────────────────────────────────────────────────────────────────

  const handleLogin = async (e) => {
    e.preventDefault()
    setError(''); setSuccess('')
    if (!loginForm.email || !loginForm.password) {
      setError('Veuillez remplir tous les champs.'); return
    }
    setLoading(true)
    try {
      const res = await api.post('/auth/login', {
        email: loginForm.email,
        password: loginForm.password,
      })
      // Backend wraps response in { success, data: { accessToken, refreshToken, user } }
      const payload = res.data?.data ?? res.data
      const { accessToken, refreshToken, user } = payload

      setAuth(user, accessToken, refreshToken)

      if (user.role === 'CLIENT') navigate('/client/dashboard')
      else navigate('/investigator/dashboard')
    } catch (err) {
      const msg = err.response?.data?.message
      setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Email ou mot de passe incorrect.')
    } finally {
      setLoading(false)
    }
  }

  // ── Register ───────────────────────────────────────────────────────────────
  // Note: backend always assigns CLIENT role on register.
  // Investigator accounts are created manually by an admin.

  const handleRegister = async (e) => {
    e.preventDefault()
    setError(''); setSuccess('')
    if (!registerForm.firstName || !registerForm.lastName || !registerForm.email || !registerForm.password) {
      setError('Veuillez remplir tous les champs obligatoires.'); return
    }
    if (registerForm.password !== registerForm.confirmPassword) {
      setError('Les mots de passe ne correspondent pas.'); return
    }
    if (registerForm.password.length < 8) {
      setError('Le mot de passe doit contenir au moins 8 caracteres.'); return
    }
    setLoading(true)
    try {
      await api.post('/auth/register', {
        firstName: registerForm.firstName,
        lastName: registerForm.lastName,
        company: registerForm.company || undefined,
        email: registerForm.email,
        password: registerForm.password,
      })
      setSuccess('Compte cree avec succes ! Connectez-vous.')
      setTab('login')
      setRegisterForm({ firstName: '', lastName: '', company: '', email: '', password: '', confirmPassword: '' })
    } catch (err) {
      const msg = err.response?.data?.message
      setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la creation du compte.')
    } finally {
      setLoading(false)
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  const switchTab = (t) => { setTab(t); setError(''); setSuccess('') }

  const inputStyle = {
    width: '100%', padding: '0.72rem 0.9rem',
    border: '1.5px solid #E5E7EB', borderRadius: 6,
    fontSize: '0.9rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
    outline: 'none', backgroundColor: '#F9FAFB', boxSizing: 'border-box',
  }
  const labelStyle = {
    display: 'block', fontSize: '0.74rem', fontWeight: 600,
    textTransform: 'uppercase', letterSpacing: '0.06em',
    color: '#4B5563', marginBottom: '0.4rem',
    fontFamily: 'Helvetica Neue, Arial, sans-serif',
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div style={{ minHeight: '100vh', display: 'flex', fontFamily: 'Georgia, serif' }}>

      {/* ── LEFT PANEL ─────────────────────────────────────────────────────── */}
      <div style={{
        width: '45%',
        background: 'linear-gradient(155deg, #0F2347 0%, #1A3A6B 60%, #0D1E3D 100%)',
        display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
        padding: '3rem', position: 'relative', overflow: 'hidden',
      }}>
        {/* decorative glow */}
        <div style={{ position: 'absolute', top: 0, right: 0, width: 280, height: 280, borderRadius: '50%', background: 'radial-gradient(circle, #C9A84C, transparent)', opacity: 0.12, transform: 'translate(30%, -30%)' }} />

        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', zIndex: 1 }}>
          <div style={{ width: 42, height: 42, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>🛡</div>
          <div>
            <div style={{ color: 'white', fontWeight: 700, fontSize: '1.1rem', letterSpacing: '0.03em' }}>FraudGuard AI</div>
            <div style={{ color: '#C9A84C', fontSize: '0.68rem', letterSpacing: '0.14em', textTransform: 'uppercase', marginTop: 2 }}>Industrial Insurance</div>
          </div>
        </div>

        {/* Hero */}
        <div style={{ zIndex: 1 }}>
          <h1 style={{ color: 'white', fontSize: '2.4rem', fontWeight: 400, lineHeight: 1.25, marginBottom: '1.2rem' }}>
            Detection de fraude<br />
            <span style={{ color: '#C9A84C', fontStyle: 'italic' }}>par intelligence</span><br />
            artificielle
          </h1>
          <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: '0.92rem', lineHeight: 1.75, marginBottom: '2rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', maxWidth: 300 }}>
            Analyse multimodale des sinistres industriels — donnees capteurs, photographies et rapports traites en temps reel.
          </p>

          {/* Metrics */}
          <div style={{ display: 'flex', gap: '2rem', marginBottom: '2rem' }}>
            {[['4', 'Modeles IA'], ['<5m', 'Analyse complete'], ['80%+', 'Precision']].map(([v, l]) => (
              <div key={l}>
                <div style={{ color: '#C9A84C', fontSize: '1.8rem', fontWeight: 700, fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1 }}>{v}</div>
                <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.68rem', letterSpacing: '0.1em', textTransform: 'uppercase', marginTop: 4, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{l}</div>
              </div>
            ))}
          </div>

          {/* Feature list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
            {[
              'Isolation Forest + LSTM pour anomalies capteurs',
              'XGBoost pour classification de panne',
              'BERT multilingue pour analyse de rapports',
              'YOLOv8 pour detection de manipulation photo',
            ].map((f, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: '#C9A84C', flexShrink: 0 }} />
                <span style={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{f}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.72rem', zIndex: 1, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
          2026 FraudGuard AI — Universite M'Hamed Bougara de Boumerdes
        </div>
      </div>

      {/* ── RIGHT PANEL ────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '3rem 4rem', backgroundColor: 'white' }}>
        <div style={{ maxWidth: 420, width: '100%', margin: '0 auto' }}>

          <p style={{ fontSize: '0.72rem', letterSpacing: '0.15em', textTransform: 'uppercase', color: '#9CA3AF', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
            {tab === 'login' ? 'Bienvenue' : 'Nouveau compte'}
          </p>
          <h2 style={{ color: '#0F2347', fontSize: '2rem', fontWeight: 400, marginBottom: '0.4rem', letterSpacing: '-0.02em' }}>
            {tab === 'login' ? <><strong>Connexion</strong> a votre espace</> : <><strong>Inscription</strong></>}
          </h2>
          <p style={{ color: '#9CA3AF', fontSize: '0.86rem', marginBottom: '1.5rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.5 }}>
            {tab === 'login'
              ? 'Acces securise a votre tableau de bord de sinistres.'
              : 'Creez votre compte client pour soumettre et suivre vos sinistres.'}
          </p>

          {/* Tabs */}
          <div style={{ display: 'flex', borderBottom: '2px solid #F3F4F6', marginBottom: '1.5rem' }}>
            {[['login', 'Connexion'], ['register', 'Inscription']].map(([t, l]) => (
              <button key={t} onClick={() => switchTab(t)}
                style={{ padding: '0.6rem 1.5rem', fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', border: 'none', background: 'transparent', cursor: 'pointer', borderBottom: tab === t ? '2px solid #0F2347' : '2px solid transparent', marginBottom: -2, color: tab === t ? '#0F2347' : '#9CA3AF', fontWeight: tab === t ? 600 : 400 }}>
                {l}
              </button>
            ))}
          </div>

          {/* Alert banners */}
          {error && <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 6, padding: '0.7rem 0.9rem', color: '#C0392B', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>⚠ {error}</div>}
          {success && <div style={{ backgroundColor: '#F0FAF4', border: '1px solid #B8E4CA', borderRadius: 6, padding: '0.7rem 0.9rem', color: '#1A7A4A', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1rem' }}>✓ {success}</div>}

          {/* ── LOGIN FORM ─────────────────────────────────────────────────── */}
          {tab === 'login' && (
            <form onSubmit={handleLogin}>
              {[
                { label: 'Adresse email', name: 'email', type: 'email', placeholder: 'votre@email.com' },
                { label: 'Mot de passe', name: 'password', type: 'password', placeholder: '••••••••' },
              ].map(f => (
                <div key={f.name} style={{ marginBottom: '1rem' }}>
                  <label style={labelStyle}>{f.label}</label>
                  <input
                    type={f.type} placeholder={f.placeholder}
                    value={loginForm[f.name]}
                    onChange={e => setLoginForm({ ...loginForm, [f.name]: e.target.value })}
                    style={inputStyle}
                    onFocus={e => e.target.style.borderColor = '#0F2347'}
                    onBlur={e => e.target.style.borderColor = '#E5E7EB'}
                  />
                </div>
              ))}
              <button type="submit" disabled={loading}
                style={{ width: '100%', padding: '0.85rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 6, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 15px rgba(15,35,71,0.25)', marginTop: '0.5rem' }}>
                {loading ? 'Connexion en cours...' : 'Se connecter'}
              </button>

              {/* Info banner: investigators cannot self-register */}
              <div style={{ marginTop: '1rem', padding: '0.6rem 0.85rem', backgroundColor: '#EBF5FB', border: '1px solid #AED6F1', borderRadius: 6, fontSize: '0.75rem', color: '#1A5276', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                ℹ Les comptes investigateurs sont crees par l'administrateur.
              </div>
            </form>
          )}

          {/* ── REGISTER FORM ──────────────────────────────────────────────── */}
          {tab === 'register' && (
            <form onSubmit={handleRegister}>

              {/* First + Last name */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.8rem' }}>
                <div>
                  <label style={labelStyle}>Prenom *</label>
                  <input type="text" placeholder="Votre prenom" value={registerForm.firstName}
                    onChange={e => setRegisterForm({ ...registerForm, firstName: e.target.value })}
                    style={inputStyle}
                    onFocus={e => e.target.style.borderColor = '#0F2347'}
                    onBlur={e => e.target.style.borderColor = '#E5E7EB'} />
                </div>
                <div>
                  <label style={labelStyle}>Nom *</label>
                  <input type="text" placeholder="Votre nom" value={registerForm.lastName}
                    onChange={e => setRegisterForm({ ...registerForm, lastName: e.target.value })}
                    style={inputStyle}
                    onFocus={e => e.target.style.borderColor = '#0F2347'}
                    onBlur={e => e.target.style.borderColor = '#E5E7EB'} />
                </div>
              </div>

              {/* Other fields */}
              {[
                { label: 'Entreprise', name: 'company', type: 'text', placeholder: 'Nom de votre entreprise (optionnel)' },
                { label: 'Adresse email *', name: 'email', type: 'email', placeholder: 'votre@email.com' },
                { label: 'Mot de passe * (maj + min + chiffre)', name: 'password', type: 'password', placeholder: 'Ex: MonMotDePasse1' },
                { label: 'Confirmer le mot de passe *', name: 'confirmPassword', type: 'password', placeholder: 'Repetez le mot de passe' },
              ].map(f => (
                <div key={f.name} style={{ marginBottom: '0.8rem' }}>
                  <label style={labelStyle}>{f.label}</label>
                  <input type={f.type} placeholder={f.placeholder} value={registerForm[f.name]}
                    onChange={e => setRegisterForm({ ...registerForm, [f.name]: e.target.value })}
                    style={inputStyle}
                    onFocus={e => e.target.style.borderColor = '#0F2347'}
                    onBlur={e => e.target.style.borderColor = '#E5E7EB'} />
                </div>
              ))}

              <button type="submit" disabled={loading}
                style={{ width: '100%', padding: '0.85rem', marginTop: '0.5rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 6, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 15px rgba(15,35,71,0.25)' }}>
                {loading ? 'Creation du compte...' : 'Creer mon compte client'}
              </button>
            </form>
          )}

          {/* Tab switch link */}
          <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
            {tab === 'login' ? 'Pas encore de compte ? ' : 'Deja un compte ? '}
            <span onClick={() => switchTab(tab === 'login' ? 'register' : 'login')}
              style={{ color: '#0F2347', fontWeight: 600, cursor: 'pointer', textDecoration: 'underline' }}>
              {tab === 'login' ? "S'inscrire" : 'Se connecter'}
            </span>
          </p>

          {/* Trust badges */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', marginTop: '1.5rem', paddingTop: '1.25rem', borderTop: '1px solid #F3F4F6' }}>
            {['JWT Securise', 'HTTPS', 'bcrypt'].map(label => (
              <div key={label} style={{ color: '#9CA3AF', fontSize: '0.72rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>{label}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}