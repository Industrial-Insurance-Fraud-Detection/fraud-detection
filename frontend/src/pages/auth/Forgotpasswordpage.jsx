import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'
import AuthLeftPanel from './AuthLeftPanel'

export default function ForgotPasswordPage() {
    const navigate = useNavigate()
    const [email, setEmail] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')

    const inp = {
        width: '100%', padding: '0.75rem 1rem',
        border: '1.5px solid #E5E7EB', borderRadius: 8,
        fontSize: '0.88rem', fontFamily: 'Helvetica Neue,Arial,sans-serif',
        outline: 'none', backgroundColor: '#F9FAFB', boxSizing: 'border-box',
        transition: 'border-color 0.2s, box-shadow 0.2s', color: '#0F2347',
    }
    const onFocus = e => { e.target.style.borderColor = '#C9A84C'; e.target.style.boxShadow = '0 0 0 3px rgba(201,168,76,0.12)' }
    const onBlur = e => { e.target.style.borderColor = '#E5E7EB'; e.target.style.boxShadow = 'none' }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        if (!email.trim()) { setError('Veuillez saisir votre adresse email.'); return }
        setLoading(true)
        try {
            const res = await api.post('/auth/forgot-password', { email })
            const data = res.data?.data ?? res.data
            // Navigate directly to reset page, passing token if available (dev mode)
            navigate('/reset-password', { state: { token: data.resetToken || '' } })
        } catch (err) {
            const msg = err.response?.data?.message
            setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la demande.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div style={{ minHeight: '100vh', display: 'flex', fontFamily: 'Georgia,serif' }}>

            <AuthLeftPanel />

            {/* RIGHT */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '2.5rem 4rem', backgroundColor: 'white' }}>
                <div style={{ maxWidth: 420, width: '100%', margin: '0 auto' }}>

                    {/* Back link */}
                    <button onClick={() => navigate('/login')} style={{
                        background: 'none', border: 'none', color: '#9CA3AF', fontSize: '0.8rem',
                        cursor: 'pointer', fontFamily: 'Helvetica Neue,Arial,sans-serif',
                        padding: 0, marginBottom: '1.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem',
                    }}>
                        ← Retour à la connexion
                    </button>

                    <p style={{ fontSize: '0.68rem', letterSpacing: '0.16em', textTransform: 'uppercase', color: '#9CA3AF', marginBottom: '0.35rem', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>
                        Sécurité
                    </p>
                    <h2 style={{ color: '#0F2347', fontSize: '1.85rem', fontWeight: 400, marginBottom: '0.35rem', letterSpacing: '-0.02em' }}>
                        <strong>Mot de passe</strong> oublié
                    </h2>
                    <p style={{ color: '#9CA3AF', fontSize: '0.83rem', marginBottom: '2rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', lineHeight: 1.55 }}>
                        Entrez votre adresse email. Vous serez redirigé pour choisir un nouveau mot de passe.
                    </p>

                    {error && (
                        <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 8, padding: '0.65rem 0.9rem', color: '#C0392B', fontSize: '0.81rem', fontFamily: 'Helvetica Neue,Arial,sans-serif', marginBottom: '1.25rem', display: 'flex', gap: '0.5rem' }}>
                            <span>⚠</span> {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit}>
                        <div style={{ marginBottom: '1.25rem' }}>
                            <label style={{ display: 'block', fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#6B7280', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>
                                Adresse email
                            </label>
                            <input type="email" placeholder="votre@email.com" value={email}
                                onChange={e => setEmail(e.target.value)}
                                style={inp} onFocus={onFocus} onBlur={onBlur} />
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
                            {loading ? 'Envoi en cours…' : 'Continuer →'}
                        </button>
                    </form>

                    <p style={{ textAlign: 'center', marginTop: '1.25rem', fontSize: '0.79rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue,Arial,sans-serif' }}>
                        Vous vous souvenez de votre mot de passe ?{' '}
                        <span onClick={() => navigate('/login')} style={{ color: '#C9A84C', fontWeight: 600, cursor: 'pointer' }}>
                            Se connecter
                        </span>
                    </p>
                </div>
            </div>
        </div>
    )
}