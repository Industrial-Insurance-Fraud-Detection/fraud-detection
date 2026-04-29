import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../../api/axios'

/**
 * ForgotPasswordPage
 * POST /auth/forgot-password  →  { message, resetToken? }
 * In dev mode the backend returns resetToken directly in the response.
 * We display it so the user can copy it to use on the reset page.
 */
export default function ForgotPasswordPage() {
    const navigate = useNavigate()
    const [email, setEmail] = useState('')
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [resetToken, setResetToken] = useState('')
    const [submitted, setSubmitted] = useState(false)

    const handleSubmit = async (e) => {
        e.preventDefault()
        setError('')
        if (!email.trim()) { setError('Veuillez saisir votre adresse email.'); return }
        setLoading(true)
        try {
            const res = await api.post('/auth/forgot-password', { email })
            const data = res.data?.data ?? res.data
            setResetToken(data.resetToken || '')
            setSubmitted(true)
        } catch (err) {
            const msg = err.response?.data?.message
            setError(Array.isArray(msg) ? msg.join(', ') : msg || 'Erreur lors de la demande.')
        } finally {
            setLoading(false)
        }
    }

    const inputStyle = {
        width: '100%', padding: '0.75rem 1rem',
        border: '1.5px solid #E5E7EB', borderRadius: 8,
        fontSize: '0.92rem', fontFamily: 'Helvetica Neue, Arial, sans-serif',
        outline: 'none', backgroundColor: '#F9FAFB',
        boxSizing: 'border-box', transition: 'border-color 0.2s',
    }

    return (
        <div style={{ minHeight: '100vh', display: 'flex', fontFamily: 'Georgia, serif', backgroundColor: '#F7F8FC' }}>

            {/* Left panel */}
            <div style={{
                width: '45%',
                background: 'linear-gradient(155deg, #0F2347 0%, #1A3A6B 60%, #0D1E3D 100%)',
                display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
                padding: '3rem', position: 'relative', overflow: 'hidden',
            }}>
                <div style={{ position: 'absolute', top: 0, right: 0, width: 280, height: 280, borderRadius: '50%', background: 'radial-gradient(circle, #C9A84C, transparent)', opacity: 0.12, transform: 'translate(30%, -30%)' }} />
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', zIndex: 1 }}>
                    <div style={{ width: 42, height: 42, borderRadius: 8, background: 'linear-gradient(135deg, #C9A84C, #E8C97A)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>🛡</div>
                    <div>
                        <div style={{ color: 'white', fontWeight: 700, fontSize: '1.1rem' }}>FraudGuard AI</div>
                        <div style={{ color: '#C9A84C', fontSize: '0.68rem', letterSpacing: '0.14em', textTransform: 'uppercase' }}>Industrial Insurance</div>
                    </div>
                </div>
                <div style={{ zIndex: 1 }}>
                    <div style={{ fontSize: '3.5rem', marginBottom: '1rem' }}>🔐</div>
                    <h1 style={{ color: 'white', fontSize: '2rem', fontWeight: 400, lineHeight: 1.3, marginBottom: '1rem' }}>
                        Récupération<br />
                        <span style={{ color: '#C9A84C', fontStyle: 'italic' }}>de mot de passe</span>
                    </h1>
                    <p style={{ color: 'rgba(255,255,255,0.55)', fontSize: '0.9rem', lineHeight: 1.7, fontFamily: 'Helvetica Neue, Arial, sans-serif', maxWidth: 300 }}>
                        Saisissez votre adresse email pour recevoir un jeton de réinitialisation de mot de passe.
                    </p>
                </div>
                <div style={{ color: 'rgba(255,255,255,0.25)', fontSize: '0.72rem', zIndex: 1, fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                    2026 FraudGuard AI
                </div>
            </div>

            {/* Right panel */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '3rem 4rem', backgroundColor: 'white' }}>
                <div style={{ maxWidth: 420, width: '100%', margin: '0 auto' }}>

                    <button onClick={() => navigate('/login')}
                        style={{ background: 'none', border: 'none', color: '#9CA3AF', fontSize: '0.82rem', cursor: 'pointer', fontFamily: 'Helvetica Neue, Arial, sans-serif', padding: 0, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        ← Retour à la connexion
                    </button>

                    <p style={{ fontSize: '0.72rem', letterSpacing: '0.15em', textTransform: 'uppercase', color: '#9CA3AF', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                        Sécurité
                    </p>
                    <h2 style={{ color: '#0F2347', fontSize: '2rem', fontWeight: 400, marginBottom: '0.5rem' }}>
                        <strong>Mot de passe</strong> oublié
                    </h2>
                    <p style={{ color: '#9CA3AF', fontSize: '0.86rem', marginBottom: '2rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', lineHeight: 1.5 }}>
                        Entrez votre email ci-dessous pour recevoir votre jeton de réinitialisation.
                    </p>

                    {error && (
                        <div style={{ backgroundColor: '#FDF2F2', border: '1px solid #EBCECE', borderRadius: 8, padding: '0.75rem 1rem', color: '#C0392B', fontSize: '0.82rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', marginBottom: '1.25rem' }}>
                            ⚠ {error}
                        </div>
                    )}

                    {!submitted ? (
                        <form onSubmit={handleSubmit}>
                            <div style={{ marginBottom: '1.25rem' }}>
                                <label style={{ display: 'block', fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#4B5563', marginBottom: '0.4rem', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                    Adresse email
                                </label>
                                <input
                                    type="email"
                                    placeholder="votre@email.com"
                                    value={email}
                                    onChange={e => setEmail(e.target.value)}
                                    style={inputStyle}
                                    onFocus={e => e.target.style.borderColor = '#0F2347'}
                                    onBlur={e => e.target.style.borderColor = '#E5E7EB'}
                                />
                            </div>
                            <button type="submit" disabled={loading}
                                style={{ width: '100%', padding: '0.85rem', background: 'linear-gradient(135deg, #0F2347, #1A3A6B)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1, boxShadow: '0 4px 15px rgba(15,35,71,0.25)' }}>
                                {loading ? 'Envoi en cours...' : 'Envoyer le jeton'}
                            </button>
                        </form>
                    ) : (
                        <div>
                            {/* Success state */}
                            <div style={{ backgroundColor: '#F0FAF4', border: '1px solid #B8E4CA', borderRadius: 10, padding: '1.25rem', marginBottom: '1.25rem' }}>
                                <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>✓</div>
                                <div style={{ fontWeight: 600, color: '#1A7A4A', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.9rem', marginBottom: '0.3rem' }}>
                                    Demande reçue !
                                </div>
                                <div style={{ color: '#4B5563', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontSize: '0.82rem', lineHeight: 1.5 }}>
                                    Si cette adresse email est enregistrée, un jeton a été généré.
                                </div>
                            </div>

                            {/* Dev mode: show reset token */}
                            {resetToken && (
                                <div style={{ backgroundColor: '#EBF5FB', border: '1px solid #AED6F1', borderRadius: 10, padding: '1.25rem', marginBottom: '1.25rem' }}>
                                    <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.1em', color: '#1A5276', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, marginBottom: '0.5rem' }}>
                                        ℹ Mode développement — Jeton de réinitialisation
                                    </div>
                                    <div style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: '#0F2347', backgroundColor: 'white', padding: '0.6rem 0.8rem', borderRadius: 6, border: '1px solid #AED6F1', wordBreak: 'break-all', marginBottom: '0.75rem' }}>
                                        {resetToken}
                                    </div>
                                    <div style={{ fontSize: '0.72rem', color: '#1A5276', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                                        En production, ce jeton serait envoyé par email. Copiez-le pour l'utiliser sur la page de réinitialisation.
                                    </div>
                                </div>
                            )}

                            <button
                                onClick={() => navigate('/reset-password', { state: { token: resetToken } })}
                                style={{ width: '100%', padding: '0.85rem', background: 'linear-gradient(135deg, #1A7A4A, #27AE60)', color: 'white', border: 'none', borderRadius: 8, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', fontWeight: 600, cursor: 'pointer', marginBottom: '0.75rem' }}>
                                Réinitialiser le mot de passe →
                            </button>

                            <button onClick={() => { setSubmitted(false); setResetToken(''); setEmail('') }}
                                style={{ width: '100%', padding: '0.75rem', border: '1.5px solid #E5E7EB', borderRadius: 8, fontSize: '0.86rem', fontFamily: 'Helvetica Neue, Arial, sans-serif', cursor: 'pointer', background: 'white', color: '#6B7280' }}>
                                Saisir un autre email
                            </button>
                        </div>
                    )}

                    <p style={{ textAlign: 'center', marginTop: '1.5rem', fontSize: '0.8rem', color: '#9CA3AF', fontFamily: 'Helvetica Neue, Arial, sans-serif' }}>
                        Vous souvenez-vous de votre mot de passe ?{' '}
                        <span onClick={() => navigate('/login')}
                            style={{ color: '#0F2347', fontWeight: 600, cursor: 'pointer', textDecoration: 'underline' }}>
                            Se connecter
                        </span>
                    </p>
                </div>
            </div>
        </div>
    )
}